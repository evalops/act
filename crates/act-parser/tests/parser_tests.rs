use act_parser::parse_module;
use act_syntax::ast::*;

fn parse(src: &str) -> Module {
    parse_module(src, 1).unwrap_or_else(|e| panic!("parse failed: {} at {:?}", e.message, e.span))
}

#[test]
fn parses_if_else_chain() {
    let m = parse(
        r#"
module test@0.1
task t(n: Int) -> Int {
  if n > 0 {
    return ok(1)
  } else if n < 0 {
    return ok(2)
  } else {
    return ok(0)
  }
}
"#,
    );
    let task = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Task(d) => Some(d.as_ref()),
            _ => None,
        })
        .unwrap();
    // body: [If]
    assert_eq!(task.body.as_ref().unwrap().len(), 1);
}

#[test]
fn parens_preserve_precedence() {
    let m = parse(
        r#"
module test@0.1
task t(a: Int, b: Int, c: Int) -> Int {
  let x = (a + b) * c
  return ok(x)
}
"#,
    );
    let task = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Task(d) => Some(d.as_ref()),
            _ => None,
        })
        .unwrap();
    let stmt = &task.body.as_ref().unwrap()[0];
    let init = match &stmt.node {
        Stmt::Let { init, .. } => init,
        _ => unreachable!(),
    };
    // Top-level op must be Mul (parens forced add into the lhs).
    match &init.node {
        Expr::Bin {
            op: BinOp::Mul,
            lhs,
            ..
        } => match &lhs.node {
            Expr::Bin { op: BinOp::Add, .. } => {}
            other => panic!("expected nested Add, got {:?}", other),
        },
        other => panic!("expected Mul, got {:?}", other),
    }
}

#[test]
fn soft_keyword_named_args() {
    // `value` and `input` are soft keywords; they must work as arg names.
    let m = parse(
        r#"
module test@0.1
task t() -> Int {
  let r = state.update(key: "k", value: 1, input: 2, expected_version: 3)
  return ok(0)
}
"#,
    );
    let task = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Task(d) => Some(d.as_ref()),
            _ => None,
        })
        .unwrap();
    let stmt = &task.body.as_ref().unwrap()[0];
    let init = match &stmt.node {
        Stmt::Let { init, .. } => init,
        _ => unreachable!(),
    };
    let args = match &init.node {
        Expr::Call { args, .. } => args,
        _ => unreachable!(),
    };
    assert_eq!(args.len(), 4);
    assert!(args.iter().all(|a| a.name.is_some()));
}

#[test]
fn empty_record_and_array() {
    let m = parse(
        r#"
module test@0.1
task t() -> Int {
  let r = {}
  let a = []
  return ok(0)
}
"#,
    );
    let task = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Task(d) => Some(d.as_ref()),
            _ => None,
        })
        .unwrap();
    let body = task.body.as_ref().unwrap();
    assert!(
        matches!(&body[0].node, Stmt::Let { init, .. } if matches!(init.node, Expr::Record(_)))
    );
    assert!(matches!(&body[1].node, Stmt::Let { init, .. } if matches!(init.node, Expr::Array(_))));
}

#[test]
fn parses_all_type_bodies() {
    let m = parse(
        r#"
module test@0.1

type Alias = String

type Rec = {
  a: String,
  b?: Int,
}

type Sum =
  | ok(value: Int)
  | err(msg: String)
  | empty

type Refined = Int where self > 0

type Op
"#,
    );
    let names: Vec<_> = m
        .items
        .iter()
        .filter_map(|i| match i {
            Item::TypeDecl(t) => Some(t.name.node.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["Alias", "Rec", "Sum", "Refined", "Op"]);
}

#[test]
fn parses_recover_defer_invariant() {
    let m = parse(
        r#"
module test@0.1
task t() -> Int {
  defer compensate {
    return ok(0)
  }
  invariant safety require 1 > 0
  recover Err from risky_call() {
    return ok(0)
  }
  checkpoint step ok(1) require 1 > 0
  return ok(0)
}
"#,
    );
    let task = m
        .items
        .iter()
        .find_map(|i| match i {
            Item::Task(d) => Some(d.as_ref()),
            _ => None,
        })
        .unwrap();
    let kinds: Vec<&str> = task
        .body
        .as_ref()
        .unwrap()
        .iter()
        .map(|s| match &s.node {
            Stmt::Defer { .. } => "defer",
            Stmt::Invariant { .. } => "invariant",
            Stmt::Recover { .. } => "recover",
            Stmt::Checkpoint { .. } => "checkpoint",
            Stmt::Return(_) => "return",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        ["defer", "invariant", "recover", "checkpoint", "return"]
    );
}

#[test]
fn parses_replay_expr() {
    let m = parse(
        r#"
module test@0.1
eval "e" {
  let r = replay trace("label")
  return ok(r)
}
"#,
    );
    let eval = m.items.iter().find_map(|i| match i {
        Item::Eval(t) => Some(t),
        _ => None,
    });
    assert!(eval.is_some(), "eval block should parse");
}
