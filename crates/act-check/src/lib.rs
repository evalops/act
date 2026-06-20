//! Act static checker: types, effects, capabilities, taint, budgets.
//!
//! v1 implements effect row checking and capability presence checks,
//! which are the most agent-relevant static guarantees. Type inference
//! and taint flow come next.

use act_diagnostics::{codes, Diagnostic, DiagnosticReport, Severity};
use act_syntax::ast::*;

/// Result of checking a module.
pub struct CheckOutput {
    pub report: DiagnosticReport,
    /// Resolved effect set per task/proc/fn, keyed by node id.
    pub declared_effects: Vec<(NodeId, Vec<String>)>,
}

pub fn check(module: &Module) -> CheckOutput {
    let mut diags = Vec::new();
    let mut declared = Vec::new();

    for item in &module.items {
        match item {
            Item::Fn(d) | Item::Proc(d) | Item::Task(d) => {
                check_fn(d, &mut diags);
                declared.push((
                    d.name.id,
                    d.effects
                        .iter()
                        .map(|e| path_string(&e.node.path))
                        .collect(),
                ));
            }
            Item::Agent(a) => {
                for h in &a.handlers {
                    check_block(&h.body, &a.effects, &mut diags);
                }
            }
            _ => {}
        }
    }

    CheckOutput {
        report: DiagnosticReport::new(diags),
        declared_effects: declared,
    }
}

fn path_string(p: &[Ident]) -> String {
    p.iter()
        .map(|i| i.node.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn check_fn(d: &FnDecl, diags: &mut Vec<Diagnostic>) {
    if d.body.is_none() {
        // extern-ish or declared only; skip
        return;
    }
    let declared: Vec<String> = d
        .effects
        .iter()
        .map(|e| path_string(&e.node.path))
        .collect();
    check_block(d.body.as_ref().unwrap(), &d.effects, diags);
}

/// Walk a block and verify every tool/model call's effects are declared.
fn check_block(block: &Block, declared: &[Spanned<EffectRef>], diags: &mut Vec<Diagnostic>) {
    for stmt in block {
        check_stmt(stmt, declared, diags);
    }
}

fn check_stmt(s: &Spanned<Stmt>, declared: &[Spanned<EffectRef>], diags: &mut Vec<Diagnostic>) {
    match &s.node {
        Stmt::Let { init, .. } | Stmt::Var { init, .. } => check_expr(init, declared, diags),
        Stmt::Assign { target, value } => {
            check_expr(target, declared, diags);
            check_expr(value, declared, diags);
        }
        Stmt::Expr(e) => check_expr(e, declared, diags),
        Stmt::Return(e) => {
            if let Some(e) = e {
                check_expr(e, declared, diags);
            }
        }
        Stmt::If { cond, then, else_ } => {
            check_expr(cond, declared, diags);
            check_block(then, declared, diags);
            if let Some(e) = else_ {
                check_block(e, declared, diags);
            }
        }
        Stmt::For { iter, body, .. } => {
            check_expr(iter, declared, diags);
            check_block(body, declared, diags);
        }
        Stmt::While { cond, body, max } => {
            // Unbounded effectful loop check: a `while` without `max` that
            // contains an effectful call is an error.
            if max.is_none() && block_has_effect(body) {
                diags.push(Diagnostic::new(
                    codes::E_UNBOUNDED_LOOP, Severity::Error, s.span,
                    "Effectful `while` loop has no `max` bound.",
                ).with_patch("while cond {", "while cond max 50 {").with_note("Add an explicit `max` bound, or move the loop into an `agent` event handler."));
            }
            check_expr(cond, declared, diags);
            check_block(body, declared, diags);
        }
        Stmt::Match { scrutinee, arms } => {
            check_expr(scrutinee, declared, diags);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    check_expr(g, declared, diags);
                }
                check_block(&arm.body, declared, diags);
            }
        }
        Stmt::Recover { from, body, .. } => {
            check_expr(from, declared, diags);
            check_block(body, declared, diags);
        }
        Stmt::Defer { body, .. } => check_block(body, declared, diags),
        Stmt::Require(e) | Stmt::Check(e) | Stmt::Ensure(e) => check_expr(e, declared, diags),
        Stmt::Trace { fields, .. } => {
            for (_, v) in fields {
                check_expr(v, declared, diags);
            }
        }
        Stmt::Checkpoint { body, require, .. } => {
            check_expr(body, declared, diags);
            if let Some(r) = require {
                check_expr(r, declared, diags);
            }
        }
        Stmt::Invariant { require, .. } => check_expr(require, declared, diags),
    }
}

fn block_has_effect(block: &Block) -> bool {
    block.iter().any(|s| stmt_has_effect(s))
}

fn stmt_has_effect(s: &Spanned<Stmt>) -> bool {
    match &s.node {
        Stmt::Let { init, .. } | Stmt::Var { init, .. } => expr_has_effect(init),
        Stmt::Assign { target, value } => expr_has_effect(target) || expr_has_effect(value),
        Stmt::Expr(e) => expr_has_effect(e),
        Stmt::Return(e) => e.as_ref().map_or(false, |e| expr_has_effect(e)),
        Stmt::If { cond, then, else_ } => {
            expr_has_effect(cond)
                || block_has_effect(then)
                || else_.as_ref().map_or(false, block_has_effect)
        }
        Stmt::For { iter, body, .. } => expr_has_effect(iter) || block_has_effect(body),
        Stmt::While { cond, body, .. } => expr_has_effect(cond) || block_has_effect(body),
        Stmt::Match { scrutinee, arms } => {
            expr_has_effect(scrutinee) || arms.iter().any(|a| block_has_effect(&a.body))
        }
        Stmt::Recover { from, body, .. } => expr_has_effect(from) || block_has_effect(body),
        Stmt::Defer { body, .. } => block_has_effect(body),
        Stmt::Require(e) | Stmt::Check(e) | Stmt::Ensure(e) => expr_has_effect(e),
        Stmt::Trace { fields, .. } => fields.iter().any(|(_, v)| expr_has_effect(v)),
        Stmt::Checkpoint { body, require, .. } => {
            expr_has_effect(body) || require.as_ref().map_or(false, expr_has_effect)
        }
        Stmt::Invariant { require, .. } => expr_has_effect(require),
    }
}

fn expr_has_effect(e: &Spanned<Expr>) -> bool {
    match &e.node {
        Expr::Lit(_) => false,
        Expr::Path(_) => false,
        Expr::Interp(parts) | Expr::Markdown(parts) => parts
            .iter()
            .any(|p| matches!(p, InterpPart::Expr(e) if expr_has_effect(e))),
        Expr::Call { callee, args } => {
            // Calls to declared tools/models are effectful.
            is_tool_or_model_callee(callee)
                || callee.uses_effect()
                || args.iter().any(|a| expr_has_effect(&a.value))
        }
        Expr::Method { receiver, args, .. } => {
            expr_has_effect(receiver) || args.iter().any(|a| expr_has_effect(&a.value))
        }
        Expr::Field { receiver, .. } => expr_has_effect(receiver),
        Expr::Index { receiver, index } => expr_has_effect(receiver) || expr_has_effect(index),
        Expr::Bin { lhs, rhs, .. } => expr_has_effect(lhs) || expr_has_effect(rhs),
        Expr::Un { expr, .. } => expr_has_effect(expr),
        Expr::Try(e) => expr_has_effect(e),
        Expr::Await(_, body) => await_body_has_effect(body),
        Expr::Infer { .. } => true,
        Expr::Decide { source, .. } => expr_has_effect(source),
        Expr::ResultCtor { value, .. } => value.as_ref().map_or(false, |v| expr_has_effect(v)),
        Expr::Spawn { .. } => true,
        Expr::Hole(_) => false,
        Expr::Record(fields) => fields.iter().any(|(_, v)| expr_has_effect(v)),
        Expr::Array(elems) => elems.iter().any(expr_has_effect),
        Expr::Block(b) => block_has_effect(b),
        Expr::ParallelRecord(fields) => fields.iter().any(|(_, v)| expr_has_effect(v)),
    }
}

fn await_body_has_effect(b: &Spanned<AwaitBody>) -> bool {
    match &b.node {
        AwaitBody::All(branches) => branches.iter().any(|(_, e)| expr_has_effect(e)),
        AwaitBody::Map { iter, body, .. } => expr_has_effect(iter) || block_has_effect(body),
        AwaitBody::Race { branches, .. } => branches.iter().any(|(_, e)| expr_has_effect(e)),
        AwaitBody::Quorum { branches, .. } => branches.iter().any(|(_, e)| expr_has_effect(e)),
    }
}

/// A callee is a tool/model call if its path has at least two segments
/// (e.g. `gh.create_pull_request`, `eo.fetch_logs`). Bare functions like
/// `rank_attempts` are not effectful unless they declare effects (future).
fn is_tool_or_model_callee(callee: &Spanned<Expr>) -> bool {
    if let Expr::Path(p) = &callee.node {
        p.len() >= 2
    } else {
        false
    }
}

trait EffectCallee {
    fn uses_effect(&self) -> bool;
}
impl EffectCallee for Spanned<Expr> {
    fn uses_effect(&self) -> bool {
        false
    }
}

fn check_expr(e: &Spanned<Expr>, declared: &[Spanned<EffectRef>], diags: &mut Vec<Diagnostic>) {
    match &e.node {
        Expr::Call { callee, args } => {
            // Determine required effect from callee path.
            if let Expr::Path(path) = &callee.node {
                if path.len() >= 2 {
                    let effect = format!("{}.{}", path[0].node, "write".to_string()); // simplified
                                                                                      // Heuristic: if the tool path ends with create_/update_/delete_/close_/merge_/push_ etc -> write; else read.
                    let last = path.last().unwrap().node.as_str();
                    let access = if is_write_name(last) { "write" } else { "read" };
                    let required = format!("{}.{}", path[0].node, access);
                    if !declared.iter().any(|d| {
                        path_string(&d.node.path) == required || path_string(&d.node.path) == path[0].node
                    }) {
                        let declared_str = declared
                            .iter()
                            .map(|d| path_string(&d.node.path))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let effects_decl = if declared.is_empty() {
                            format!("effects []")
                        } else {
                            format!("effects [{}]", declared_str)
                        };
                        diags.push(Diagnostic::new(
                            codes::E_EFFECT_MISSING, Severity::Error, e.span,
                            format!("Call to `{}` requires effect `{}`, but the enclosing scope does not declare it.", path_string(path), required),
                        ).with_expected(format!("effects include {}", required))
                         .with_actual(effects_decl.clone())
                         .with_patch(effects_decl, format!("effects [{}, {}]", declared_str, required)));
                    }
                }
            }
            check_expr(callee, declared, diags);
            for a in args {
                check_expr(&a.value, declared, diags);
            }
        }
        Expr::Method { receiver, args, .. } => {
            check_expr(receiver, declared, diags);
            for a in args {
                check_expr(&a.value, declared, diags);
            }
        }
        Expr::Field { receiver, .. } => check_expr(receiver, declared, diags),
        Expr::Index { receiver, index } => {
            check_expr(receiver, declared, diags);
            check_expr(index, declared, diags);
        }
        Expr::Bin { lhs, rhs, .. } => {
            check_expr(lhs, declared, diags);
            check_expr(rhs, declared, diags);
        }
        Expr::Un { expr, .. } => check_expr(expr, declared, diags),
        Expr::Try(e) => check_expr(e, declared, diags),
        Expr::Await(_, body) => check_await_body(body, declared, diags),
        Expr::Infer { model, spec, .. } => {
            check_expr(model, declared, diags);
            if !declared.iter().any(|d| path_string(&d.node.path) == "model") {
                let declared_str = declared
                    .iter()
                    .map(|d| path_string(&d.node.path))
                    .collect::<Vec<_>>()
                    .join(", ");
                let effects_decl = if declared.is_empty() {
                    "effects []".to_string()
                } else {
                    format!("effects [{}]", declared_str)
                };
                diags.push(Diagnostic::new(
                    codes::E_EFFECT_MISSING, Severity::Error, e.span,
                    "`infer` requires effect `model`, but the enclosing scope does not declare it.",
                ).with_expected("effects include model")
                 .with_actual(effects_decl.clone())
                 .with_patch(effects_decl, format!("effects [{}, model]", declared_str)));
            }
            if let Some(i) = &spec.input {
                check_expr(i, declared, diags);
            }
            for c in &spec.constraints {
                check_expr(c, declared, diags);
            }
            if let Some(g) = &spec.goal {
                check_expr(g, declared, diags);
            }
            if let Some(r) = &spec.rubric {
                check_expr(r, declared, diags);
            }
            if let Some(c) = &spec.choices {
                check_expr(c, declared, diags);
            }
            if let Some(v) = &spec.validate {
                check_expr(v, declared, diags);
            }
            if let Some(a) = &spec.accept {
                check_expr(a, declared, diags);
            }
            if let Some(b) = &spec.else_ {
                check_block(b, declared, diags);
            }
        }
        Expr::Decide {
            source,
            require,
            else_,
            ..
        } => {
            check_expr(source, declared, diags);
            if let Some(r) = require {
                check_expr(r, declared, diags);
            }
            if let Some(b) = else_ {
                check_block(b, declared, diags);
            }
        }
        Expr::ResultCtor { value, .. } => {
            if let Some(v) = value {
                check_expr(v, declared, diags);
            }
        }
        Expr::Spawn { args, caps, .. } => {
            for a in args {
                check_expr(&a.value, declared, diags);
            }
            for c in caps {
                check_expr(c, declared, diags);
            }
        }
        Expr::Hole(h) => match h {
            HoleSpec::Plain(h) => check_expr(h, declared, diags),
            HoleSpec::Constrained { goal, must_satisfy } => {
                if let Some(g) = goal {
                    check_expr(g, declared, diags);
                }
                for m in must_satisfy {
                    check_expr(m, declared, diags);
                }
            }
        },
        Expr::Record(fields) => {
            for (_, v) in fields {
                check_expr(v, declared, diags);
            }
        }
        Expr::Array(elems) => {
            for e in elems {
                check_expr(e, declared, diags);
            }
        }
        Expr::Block(b) => check_block(b, declared, diags),
        Expr::ParallelRecord(fields) => {
            for (_, v) in fields {
                check_expr(v, declared, diags);
            }
        }
        Expr::Interp(parts) | Expr::Markdown(parts) => {
            for p in parts {
                if let InterpPart::Expr(e) = p {
                    check_expr(e, declared, diags);
                }
            }
        }
        Expr::Lit(_) | Expr::Path(_) => {}
    }
}

fn check_await_body(
    b: &Spanned<AwaitBody>,
    declared: &[Spanned<EffectRef>],
    diags: &mut Vec<Diagnostic>,
) {
    match &b.node {
        AwaitBody::All(branches) => {
            for (_, e) in branches {
                check_expr(e, declared, diags);
            }
        }
        AwaitBody::Map { iter, body, .. } => {
            check_expr(iter, declared, diags);
            check_block(body, declared, diags);
        }
        AwaitBody::Race { branches, .. } => {
            for (_, e) in branches {
                check_expr(e, declared, diags);
            }
        }
        AwaitBody::Quorum { branches, .. } => {
            for (_, e) in branches {
                check_expr(e, declared, diags);
            }
        }
    }
}

fn is_write_name(name: &str) -> bool {
    let prefixes = [
        "create_",
        "update_",
        "delete_",
        "close_",
        "merge_",
        "push_",
        "apply_",
        "comment_",
        "write_",
        "run_tests",
        "run_",
    ];
    prefixes.iter().any(|p| name.starts_with(p)) || name == "create_pull_request"
}
