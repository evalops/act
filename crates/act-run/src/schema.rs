//! Render Act types to JSON Schema for structured-output model calls.
//!
//! When a task calls `infer T using ...`, the host asks the model to return
//! JSON matching `T`. Instead of describing `T` in prose (which the model can
//! ignore), we render it to a JSON Schema and pass it as `response_format:
//! { type: "json_schema", strict: true }`. The provider then *guarantees* the
//! shape server-side, eliminating silent field-drop coercion bugs.
//!
//! Only the subset of Act types that map cleanly to JSON Schema is supported.
//! Refinement types, `typeof`, and holes fall back to `{}` (any JSON), which
//! lets the run proceed without a guarantee rather than failing.

use act_syntax::ast::{Ty, TypeBody, TypeDecl};
use serde_json::{json, Map, Value as Json};

use crate::registry::TypeRegistry;

/// Render an Act type into a JSON Schema object.
pub fn ty_to_schema(ty: &Ty, types: &TypeRegistry) -> Json {
    match ty {
        Ty::Named { path, args } => {
            let name = path.last().map(|i| i.node.as_str()).unwrap_or("");
            match name {
                "String" => json!({"type": "string"}),
                "Int" => json!({"type": "integer"}),
                "Decimal" => json!({"type": "number"}),
                "Bool" => json!({"type": "boolean"}),
                "Secret" if !args.is_empty() => ty_to_schema(&args[0].node, types),
                "Result" if args.len() == 2 => {
                    // Result<T,E> coerces to { "ok": T } or { "err": E }.
                    // Use anyOf so either shape is accepted.
                    json!({
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "ok": ty_to_schema(&args[0].node, types)
                                },
                                "required": ["ok"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "err": ty_to_schema(&args[1].node, types)
                                },
                                "required": ["err"],
                                "additionalProperties": false
                            }
                        ]
                    })
                }
                other => match types.get(other) {
                    Some(decl) => decl_to_schema(decl, types),
                    None => json!({}),
                },
            }
        }
        Ty::Array(inner) => json!({
            "type": "array",
            "items": ty_to_schema(&inner.node, types)
        }),
        Ty::Tuple(elems) => {
            let items: Vec<Json> = elems.iter().map(|t| ty_to_schema(&t.node, types)).collect();
            json!({
                "type": "array",
                "prefixItems": items,
                "minItems": items.len(),
                "maxItems": items.len()
            })
        }
        _ => json!({}),
    }
}

fn decl_to_schema(decl: &TypeDecl, types: &TypeRegistry) -> Json {
    match &decl.body {
        TypeBody::Record(fields) => {
            let mut props = Map::new();
            let mut required: Vec<&str> = Vec::new();
            for f in fields {
                props.insert(f.name.node.clone(), ty_to_schema(&f.ty.node, types));
                if !f.optional {
                    required.push(f.name.node.as_str());
                }
            }
            json!({
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": false
            })
        }
        TypeBody::Enum(variants) => {
            // Sum type: anyOf { "variant_name": {fields} }.
            // For strict mode, each variant is an object with exactly one
            // property keyed by the variant name.
            let options: Vec<Json> = variants
                .iter()
                .map(|v| {
                    if v.fields.is_empty() {
                        let mut props = Map::new();
                        props.insert(v.name.node.clone(), json!({}));
                        json!({
                            "type": "object",
                            "properties": props,
                            "required": [v.name.node],
                            "additionalProperties": false
                        })
                    } else {
                        let mut inner_props = Map::new();
                        let mut inner_required: Vec<&str> = Vec::new();
                        for (n, t) in &v.fields {
                            let nm = n.as_ref().map(|i| i.node.as_str()).unwrap_or("value");
                            inner_props.insert(nm.to_string(), ty_to_schema(&t.node, types));
                            inner_required.push(nm);
                        }
                        let inner_schema = json!({
                            "type": "object",
                            "properties": inner_props,
                            "required": inner_required,
                            "additionalProperties": false
                        });
                        let mut props = Map::new();
                        props.insert(v.name.node.clone(), inner_schema);
                        json!({
                            "type": "object",
                            "properties": props,
                            "required": [v.name.node],
                            "additionalProperties": false
                        })
                    }
                })
                .collect();
            json!({"anyOf": options})
        }
        TypeBody::Alias(inner) => ty_to_schema(&inner.node, types),
        TypeBody::Refinement { ty, .. } => ty_to_schema(&ty.node, types),
        TypeBody::Opaque => json!({}),
    }
}
