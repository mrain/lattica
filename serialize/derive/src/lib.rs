use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, GenericParam, parse_macro_input, parse_quote};

/// Derive `CanonicalSerialize` for a struct.
#[proc_macro_derive(CanonicalSerialize)]
pub fn derive_canonical_serialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let mut generics = input.generics.clone();

    // Add CanonicalSerialize bound to each type parameter
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(ty_param) = param {
            ty_param
                .bounds
                .push(parse_quote!(grid_serialize::CanonicalSerialize));
        }
    }

    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    let (size_body, serialize_body) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => generate_named_fields(fields, true),
            Fields::Unnamed(fields) => generate_unnamed_fields(fields, true),
            Fields::Unit => (quote! { 0 }, quote! { Ok(()) }),
        },
        Data::Enum(_) => {
            return syn::Error::new_spanned(
                name,
                "derive(CanonicalSerialize) does not support enums yet",
            )
            .to_compile_error()
            .into();
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "derive(CanonicalSerialize) does not support unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics grid_serialize::CanonicalSerialize for #name #type_generics #where_clause {
            fn serialized_size(&self) -> usize {
                #size_body
            }

            fn serialize_into(&self, buf: &mut alloc::vec::Vec<u8>) -> Result<(), grid_serialize::SerializationError> {
                #serialize_body
                Ok(())
            }
        }
    };

    expanded.into()
}

/// Derive `CanonicalDeserialize` for a struct.
#[proc_macro_derive(CanonicalDeserialize)]
pub fn derive_canonical_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let mut generics = input.generics.clone();

    // Add CanonicalDeserialize bound to each type parameter
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(ty_param) = param {
            ty_param
                .bounds
                .push(parse_quote!(grid_serialize::CanonicalDeserialize));
        }
    }

    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();

    let (deserialize_body, _struct_literal) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => generate_named_fields(fields, false),
            Fields::Unnamed(fields) => generate_unnamed_fields(fields, false),
            Fields::Unit => (
                quote! {
                    let consumed = 0;
                    Ok((Self, consumed))
                },
                quote! { Self },
            ),
        },
        Data::Enum(_) => {
            return syn::Error::new_spanned(
                name,
                "derive(CanonicalDeserialize) does not support enums yet",
            )
            .to_compile_error()
            .into();
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "derive(CanonicalDeserialize) does not support unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics grid_serialize::CanonicalDeserialize for #name #type_generics #where_clause {
            fn deserialize(data: &[u8]) -> Result<(Self, usize), grid_serialize::SerializationError> {
                #deserialize_body
            }
        }
    };

    expanded.into()
}

fn generate_named_fields(
    fields: &syn::FieldsNamed,
    is_serialize: bool,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let field_ids: Vec<_> = fields
        .named
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();

    if is_serialize {
        let size_exprs = field_ids.iter().map(|id| {
            quote! { self.#id.serialized_size() }
        });
        let serialize_exprs = field_ids.iter().map(|id| {
            quote! {
                grid_serialize::CanonicalSerialize::serialize_into(&self.#id, buf)?;
            }
        });
        (
            quote! {
                #(#size_exprs)+*
            },
            quote! {
                #(#serialize_exprs)*
            },
        )
    } else {
        let offset_init = quote! { let mut __offset: usize = 0; };
        let mut deserialized = Vec::new();
        let mut field_assignments = Vec::new();
        for f in &fields.named {
            let ty = &f.ty;
            let id = f.ident.as_ref().unwrap();
            let var_name = format_ident!("__{}", id);
            deserialized.push(quote! {
                let (#var_name, __consumed) = <#ty as grid_serialize::CanonicalDeserialize>::deserialize(&data[__offset..])?;
                __offset += __consumed;
            });
            field_assignments.push(quote! { #id: #var_name });
        }
        let result = quote! {
            #offset_init
            #(#deserialized)*
            Ok((Self { #(#field_assignments),* }, __offset))
        };
        (result, quote! {})
    }
}

fn generate_unnamed_fields(
    fields: &syn::FieldsUnnamed,
    is_serialize: bool,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    if is_serialize {
        let indices: Vec<_> = (0..fields.unnamed.len()).collect();
        let size_exprs = indices.iter().map(|i| {
            quote! { self.#i.serialized_size() }
        });
        let serialize_exprs = indices.iter().map(|i| {
            quote! {
                grid_serialize::CanonicalSerialize::serialize_into(&self.#i, buf)?;
            }
        });
        (
            quote! {
                #(#size_exprs)+*
            },
            quote! {
                #(#serialize_exprs)*
            },
        )
    } else {
        let offset_init = quote! { let mut __offset: usize = 0; };
        let mut deserialized = Vec::new();
        let mut field_vars = Vec::new();
        for (i, f) in fields.unnamed.iter().enumerate() {
            let ty = &f.ty;
            let var_name = format_ident!("__field{}", i);
            deserialized.push(quote! {
                let (#var_name, __consumed) = <#ty as grid_serialize::CanonicalDeserialize>::deserialize(&data[__offset..])?;
                __offset += __consumed;
            });
            field_vars.push(var_name);
        }
        let result = quote! {
            #offset_init
            #(#deserialized)*
            Ok((Self (#(#field_vars),*), __offset))
        };
        (result, quote! {})
    }
}
