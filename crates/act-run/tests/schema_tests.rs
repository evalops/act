//! Tests for JSON Schema rendering from Act types.

use act_parser::parse_module;
use act_run::schema::ty_to_schema;
use act_run::TypeRegistry;
use serde_json::json;

fn registry(src: &str) -> (act_syntax::ast::Module, TypeRegistry) {
    let m = parse_module(src, 1).expect("parse");
    let reg = TypeRegistry::from_module(&m);
    (m, reg)
}

fn find_task_return_ty(m: &act_syntax::ast::Module) -> act_syntax::ast::Ty {
    m.items
        .iter()
        .find_map(|i| match i {
            act_syntax::ast::Item::Task(t) if t.name.node == "run" => {
                Some(t.return_ty.node.clone())
            }
            _ => None,
        })
        .expect("task `run` not found")
}

#[test]
fn renders_record_schema() {
    let (m, reg) = registry(
        r#"
module s@0.1
type Summary = { text: String, rating?: Decimal }
task run() -> Summary effects [model] { return ok({ text: "", rating: 0.0 }) }
"#,
    );
    let ty = find_task_return_ty(&m);
    let schema = ty_to_schema(&ty, &reg);
    assert_eq!(
        schema,
        json!({
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "rating": {"type": "number"}
            },
            "required": ["text"],
            "additionalProperties": false
        })
    );
}

#[test]
fn renders_array_schema() {
    let (m, reg) = registry(
        r#"
module s@0.1
task run() -> [String] effects [model] { return ok([]) }
"#,
    );
    let ty = find_task_return_ty(&m);
    let schema = ty_to_schema(&ty, &reg);
    assert_eq!(
        schema,
        json!({
            "type": "array",
            "items": {"type": "string"}
        })
    );
}

#[test]
fn renders_primitives() {
    let (m, _reg) = registry(
        r#"
module s@0.1
task run() -> String effects [model] { return ok("") }
"#,
    );
    let ty = find_task_return_ty(&m);
    let schema = ty_to_schema(&ty, &TypeRegistry::from_module(&m));
    assert_eq!(schema, json!({"type": "string"}));
}

#[test]
fn renders_result_schema() {
    let (m, reg) = registry(
        r#"
module s@0.1
task run() -> Result<String, String> effects [model] { return ok("") }
"#,
    );
    let ty = find_task_return_ty(&m);
    let schema = ty_to_schema(&ty, &reg);
    assert!(schema.get("anyOf").is_some(), "expected anyOf for Result");
}

#[test]
fn renders_enum_schema() {
    let (m, reg) = registry(
        r#"
module s@0.1
type Status = | ok | err(msg: String)
task run() -> Status effects [model] { return ok(ok) }
"#,
    );
    let ty = find_task_return_ty(&m);
    let schema = ty_to_schema(&ty, &reg);
    let any_of = schema
        .get("anyOf")
        .and_then(|a| a.as_array())
        .expect("expected anyOf for enum");
    assert_eq!(any_of.len(), 2);
}
