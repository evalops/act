//! Runtime values for Act.

use act_syntax::ast::{Literal, Ty, TypeBody, TypeDecl};
use serde_json::Value as Json;

use crate::registry::TypeRegistry;

/// A runtime value.
#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Decimal(f64),
    String(String),
    Bool(bool),
    Null,
    Array(Vec<Value>),
    /// Ordered record literal / coerced object.
    Record(Vec<(String, Value)>),
    /// `ok(v)` / `err(e)`.
    Result {
        ok: bool,
        value: Option<Box<Value>>,
    },
    /// `Secret<T>` — opaque until redacted.
    Secret(Box<Value>),
}

impl Value {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn truthy(&self) -> bool {
        self.as_bool().unwrap_or(false)
    }

    /// Ordered field lookup on a record (linear scan; records are small).
    pub fn field(&self, name: &str) -> Option<&Value> {
        if let Value::Record(fields) = self {
            fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
        } else {
            None
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Decimal(d) => Some(*d),
            _ => None,
        }
    }
}

/// Coerce a JSON value into a runtime `Value` guided by a declared type.
///
/// Model output arrives as JSON; we use the task's declared type so the
/// interpreter sees properly-shaped values (records, arrays, results).
pub fn coerce(ty: &Ty, json: &Json, types: &TypeRegistry) -> Result<Value, String> {
    Ok(match ty {
        Ty::Named { path, args } => {
            let name = path.last().map(|i| i.node.as_str()).unwrap_or("");
            match name {
                "Int" => Value::Int(json.as_i64().ok_or("expected int")?),
                "Decimal" => Value::Decimal(
                    json.as_f64()
                        .or_else(|| json.as_str().and_then(|s| s.parse().ok()))
                        .ok_or("expected decimal")?,
                ),
                "String" => Value::String(
                    json.as_str()
                        .map(|s| s.to_string())
                        .or_else(|| json.as_str().map(String::from))
                        .ok_or("expected string")?,
                ),
                "Bool" => Value::Bool(json.as_bool().ok_or("expected bool")?),
                "Secret" if !args.is_empty() => {
                    Value::Secret(Box::new(coerce(&args[0].node, json, types)?))
                }
                "Result" => {
                    // Result<T,E>: accept {"ok": v} / {"err": e} or a bare value as ok.
                    if let Some(obj) = json.as_object() {
                        if let Some(v) = obj.get("ok") {
                            let inner = if v.is_null() {
                                None
                            } else {
                                Some(Box::new(coerce(&args[0].node, v, types)?))
                            };
                            return Ok(Value::Result {
                                ok: true,
                                value: inner,
                            });
                        }
                        if let Some(v) = obj.get("err") {
                            let inner = if v.is_null() {
                                None
                            } else {
                                Some(Box::new(coerce(&args[1].node, v, types)?))
                            };
                            return Ok(Value::Result {
                                ok: false,
                                value: inner,
                            });
                        }
                    }
                    let inner = Some(Box::new(coerce(&args[0].node, json, types)?));
                    Value::Result {
                        ok: true,
                        value: inner,
                    }
                }
                other => match types.get(other) {
                    Some(decl) => coerce_decl(decl, json, types)?,
                    None => coerce_fallback(json),
                },
            }
        }
        Ty::Array(inner) => {
            let arr = json
                .as_array()
                .ok_or_else(|| format!("expected array, got {}", json))?;
            let mut out = Vec::with_capacity(arr.len());
            for e in arr {
                out.push(coerce(&inner.node, e, types)?);
            }
            Value::Array(out)
        }
        Ty::Tuple(elems) => {
            let arr = json.as_array().ok_or("expected tuple as array")?;
            let mut out = Vec::with_capacity(elems.len());
            for (i, ty) in elems.iter().enumerate() {
                out.push(coerce(&ty.node, arr.get(i).unwrap_or(&Json::Null), types)?);
            }
            Value::Record(
                out.into_iter()
                    .enumerate()
                    .map(|(i, v)| (format!("_{}", i), v))
                    .collect(),
            )
        }
        _ => coerce_fallback(json),
    })
}

fn coerce_decl(decl: &TypeDecl, json: &Json, types: &TypeRegistry) -> Result<Value, String> {
    match &decl.body {
        TypeBody::Record(fields) => {
            let obj = json
                .as_object()
                .ok_or_else(|| format!("expected object for {}", decl.name.node))?;
            let mut out = Vec::with_capacity(fields.len());
            for f in fields {
                let raw = obj.get(&f.name.node).unwrap_or(&Json::Null);
                if f.optional && raw.is_null() {
                    continue;
                }
                out.push((f.name.node.clone(), coerce(&f.ty.node, raw, types)?));
            }
            Ok(Value::Record(out))
        }
        TypeBody::Enum(variants) => {
            // A variant as {"variant_name": {fields}} or a bare string tag.
            if let Some(obj) = json.as_object() {
                for v in variants {
                    if let Some(payload) = obj.get(&v.name.node) {
                        let inner = if v.fields.is_empty() || payload.is_null() {
                            None
                        } else if let Some((_, fty)) = v.fields.first() {
                            Some(Box::new(coerce(&fty.node, payload, types)?))
                        } else {
                            None
                        };
                        return Ok(Value::Result {
                            ok: v.name.node == "ok",
                            value: inner,
                        });
                    }
                }
            }
            if let Some(tag) = json.as_str() {
                for v in variants {
                    if v.name.node == tag {
                        return Ok(Value::Result {
                            ok: tag == "ok",
                            value: None,
                        });
                    }
                }
            }
            Err(format!("no enum variant matches {}", json))
        }
        TypeBody::Alias(inner) => coerce(&inner.node, json, types),
        TypeBody::Refinement { ty, .. } => coerce(&ty.node, json, types),
        TypeBody::Opaque => Ok(coerce_fallback(json)),
    }
}

fn coerce_fallback(json: &Json) -> Value {
    value_from_json(json)
}

/// Convert a JSON value into a runtime `Value` without type guidance
/// (used for CLI args and host-supplied inputs).
pub fn value_from_json(json: &Json) -> Value {
    match json {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) => n
            .as_i64()
            .map(Value::Int)
            .or_else(|| n.as_f64().map(Value::Decimal))
            .unwrap_or(Value::Null),
        Json::String(s) => Value::String(s.clone()),
        Json::Array(a) => Value::Array(a.iter().map(value_from_json).collect()),
        Json::Object(o) => Value::Record(
            o.iter()
                .map(|(k, v)| (k.clone(), value_from_json(v)))
                .collect(),
        ),
    }
}

/// Convert a runtime value into JSON for host payloads (model input, args).
pub fn to_json(v: &Value) -> Json {
    match v {
        Value::Int(n) => Json::Number((*n).into()),
        Value::Decimal(d) => serde_json::Number::from_f64(*d)
            .map(Json::Number)
            .unwrap_or(Json::Null),
        Value::String(s) => Json::String(s.clone()),
        Value::Bool(b) => Json::Bool(*b),
        Value::Null => Json::Null,
        Value::Array(a) => Json::Array(a.iter().map(to_json).collect()),
        Value::Record(fs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in fs {
                map.insert(k.clone(), to_json(v));
            }
            Json::Object(map)
        }
        Value::Result { ok, value } => {
            let mut map = serde_json::Map::new();
            let key = if *ok { "ok" } else { "err" };
            map.insert(
                key.to_string(),
                value.as_ref().map(|v| to_json(v)).unwrap_or(Json::Null),
            );
            Json::Object(map)
        }
        Value::Secret(inner) => to_json(inner),
    }
}

/// Materialize a literal expression value.
pub fn from_literal(l: &Literal) -> Value {
    match l {
        Literal::Int(n) => Value::Int(*n),
        Literal::Decimal(s) => Value::Decimal(s.parse().unwrap_or(0.0)),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Duration(_) | Literal::Money(_, _) | Literal::Null => Value::Null,
    }
}
