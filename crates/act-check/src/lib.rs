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

    // Collect extern tool declarations for idempotency checking.
    let tool_decls = collect_tool_decls(module);

    for item in &module.items {
        match item {
            Item::Fn(d) | Item::Proc(d) | Item::Task(d) => {
                check_fn(d, &tool_decls, &mut diags);
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

struct ToolDeclInfo {
    path: String,
    idempotent: bool,
}

fn collect_tool_decls(module: &Module) -> Vec<ToolDeclInfo> {
    let mut tools = Vec::new();
    for item in &module.items {
        if let Item::ExternTool(t) = item {
            tools.push(ToolDeclInfo {
                path: path_string(&t.path),
                idempotent: t.idempotent_by.is_some(),
            });
        }
    }
    tools
}

fn path_string(p: &[Ident]) -> String {
    p.iter()
        .map(|i| i.node.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn check_fn(d: &FnDecl, tools: &[ToolDeclInfo], diags: &mut Vec<Diagnostic>) {
    if d.body.is_none() {
        return;
    }
    let body = d.body.as_ref().unwrap();
    check_block(body, &d.effects, diags);

    // Rule: secret taint — track Secret<T> variables, reject them in model inputs.
    let mut taint_set = collect_tainted_vars(body);
    for param in &d.params {
        if type_contains_secret(&param.ty.node) {
            taint_set.insert(param.name.node.clone());
        }
    }
    check_taint(body, &taint_set, diags);

    // Rule 4: policy_expect vs needs cross-check.
    check_policy_vs_needs(d, diags);

    // Rule 3: compensation requirement for non-idempotent writes.
    if d.kind == FnKind::Task {
        let has_write = d.effects.iter().any(|e| {
            let p = path_string(&e.node.path);
            p.ends_with(".write") || p == "gh.write"
        });
        let has_budget = d.budget.is_some();
        if has_write && has_budget {
            check_compensation(body, tools, diags);
        }
    }
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
        Stmt::Require(e) | Stmt::Ensure(e) => check_expr(e, declared, diags),
        Stmt::Check { cond, else_block } => {
            check_expr(cond, declared, diags);
            // Rule 2: check without else bypasses the typed error enum.
            if else_block.is_none() {
                diags.push(
                    Diagnostic::new(
                        codes::E_CHECK_UNHANDLED,
                        Severity::Error,
                        s.span,
                        "`check` without an `else` clause bypasses the typed error enum. \
                         Add `else { return err(...) }` to map the failure to a typed error.",
                    )
                    .with_patch("check cond", "check cond else { return err(...) }"),
                );
            } else {
                check_block(else_block.as_ref().unwrap(), declared, diags);
            }
        }
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
        Stmt::Require(e) | Stmt::Check { cond: e, .. } | Stmt::Ensure(e) => expr_has_effect(e),
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
                    let _effect = format!("{}.{}", path[0].node, "write".to_string()); // simplified
                                                                                       // Heuristic: if the tool path ends with create_/update_/delete_/close_/merge_/push_ etc -> write; else read.
                    let last = path.last().unwrap().node.as_str();
                    let access = if is_write_name(last) { "write" } else { "read" };
                    let required = format!("{}.{}", path[0].node, access);
                    if !declared.iter().any(|d| {
                        path_string(&d.node.path) == required
                            || path_string(&d.node.path) == path[0].node
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
            if !declared
                .iter()
                .any(|d| path_string(&d.node.path) == "model")
            {
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
            // Rule 1: warn when model confidence threshold is unrealistically high.
            // Model self-reported confidence above 0.90 is unreliable; require
            // a verifier-derived score or explicit calibration instead.
            if let Some(accept) = &spec.accept {
                if let Some(threshold) = extract_confidence_threshold(accept) {
                    if threshold >= 0.90 {
                        diags.push(
                            Diagnostic::new(
                                codes::W_MODEL_CONFIDENCE_HIGH_THRESHOLD,
                                Severity::Warning,
                                e.span,
                                format!(
                                    "Model confidence threshold {:.2} is above 0.90. \
                                     Model self-reported confidence at this level is unreliable. \
                                     Use a VerifierScore (from tests/tools) or calibrate explicitly.",
                                    threshold
                                ),
                            ),
                        );
                    }
                }
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

// =====================================================================
// Rule 1: Confidence provenance — extract threshold from accept clause
// =====================================================================

/// Try to extract a numeric threshold from an accept expression like
/// `confidence >= 0.80` or `value.confidence >= 0.95`.
fn extract_confidence_threshold(expr: &Spanned<Expr>) -> Option<f64> {
    if let Expr::Bin {
        op: BinOp::Ge,
        lhs,
        rhs,
        ..
    } = &expr.node
    {
        // Check that lhs references "confidence"
        if expr_mentions_confidence(lhs) {
            return expr_to_f64(rhs);
        }
    }
    // Also handle `expr && expr` by recursing into the first arm.
    if let Expr::Bin {
        op: BinOp::And,
        lhs,
        ..
    } = &expr.node
    {
        return extract_confidence_threshold(lhs);
    }
    None
}

fn expr_mentions_confidence(e: &Spanned<Expr>) -> bool {
    match &e.node {
        Expr::Path(p) => p.iter().any(|i| i.node == "confidence"),
        Expr::Field { name, .. } => name.node == "confidence",
        _ => false,
    }
}

fn expr_to_f64(e: &Spanned<Expr>) -> Option<f64> {
    match &e.node {
        Expr::Lit(Literal::Decimal(s)) => s.parse().ok(),
        Expr::Lit(Literal::Int(n)) => Some(*n as f64),
        _ => None,
    }
}

// =====================================================================
// Rule 4: policy_expect vs needs — compile-time cross-check
// =====================================================================

fn check_policy_vs_needs(d: &FnDecl, diags: &mut Vec<Diagnostic>) {
    let policy = match &d.policy_expect {
        Some(p) => &p.node,
        None => return,
    };

    // Build the set of granted capability keywords.
    let granted_caps: Vec<String> = d.needs.iter().map(|c| path_string(&c.node.path)).collect();

    for clause in &policy.clauses {
        let target = path_string(&clause.target);
        match clause.verb {
            PolicyVerb::May => {
                // `may X` requires the corresponding capability to be granted.
                if !caps_cover_target(&granted_caps, &target) {
                    diags.push(
                        Diagnostic::new(
                            codes::E_POLICY_MAY_UNGRANTED,
                            Severity::Error,
                            clause.span,
                            format!(
                                "policy_expect declares `may {}` but no matching capability is granted in `needs`.",
                                target
                            ),
                        )
                        .with_note("Either add the capability to `needs`, or remove this policy clause."),
                    );
                }
            }
            PolicyVerb::MustNot => {
                // `must_not X` requires the corresponding capability to NOT be granted.
                if caps_cover_target(&granted_caps, &target) {
                    diags.push(
                        Diagnostic::new(
                            codes::E_POLICY_MUST_NOT_GRANTED,
                            Severity::Error,
                            clause.span,
                            format!(
                                "policy_expect declares `must_not {}` but a matching capability IS granted in `needs`. Contradiction.",
                                target
                            ),
                        )
                        .with_note("Remove the capability from `needs`, or remove this policy clause."),
                    );
                }
            }
            PolicyVerb::RequireHuman => {}
        }
    }
}

/// Check if any granted cap covers the policy target.
/// Requires matching the tool prefix AND at least one action verb.
/// `gh.merge_pull_request` matches `gh.pull_request.merge` (both have verb "merge"),
/// but NOT `gh.pull_request.create` (different verbs).
fn caps_cover_target(caps: &[String], target: &str) -> bool {
    let target_tool = target.split('.').next().unwrap_or("");
    let target_verbs = action_verbs(target);
    if target_verbs.is_empty() {
        return false;
    }
    caps.iter().any(|cap| {
        let cap_tool = cap.split('.').next().unwrap_or("");
        let cap_verbs = action_verbs(cap);
        cap_tool == target_tool && cap_verbs.intersection(&target_verbs).count() > 0
    })
}

/// Extract action verbs from a path. Verbs: create, merge, delete, close,
/// update, read, write, comment, push, apply, run.
fn action_verbs(s: &str) -> std::collections::HashSet<String> {
    const VERBS: &[&str] = &[
        "create", "merge", "delete", "close", "update", "read", "write", "comment", "push",
        "apply", "run",
    ];
    s.split(['.', '_'])
        .filter_map(|p| {
            let lower = p.to_lowercase();
            if VERBS.contains(&lower.as_str()) {
                Some(lower)
            } else {
                None
            }
        })
        .collect()
}

// =====================================================================
// Rule 3: Compensation requirement for non-idempotent writes
// =====================================================================

fn check_compensation(block: &Block, tools: &[ToolDeclInfo], diags: &mut Vec<Diagnostic>) {
    // If any defer compensate exists in this block, all writes are covered.
    let has_defer = block_has_defer_compensate(block);
    for stmt in block {
        check_compensation_stmt(stmt, tools, diags, has_defer);
    }
}

fn block_has_defer_compensate(block: &Block) -> bool {
    block.iter().any(|s| {
        matches!(
            &s.node,
            Stmt::Defer {
                kind: DeferKind::Compensate,
                ..
            }
        )
    })
}

fn check_compensation_stmt(
    s: &Spanned<Stmt>,
    tools: &[ToolDeclInfo],
    diags: &mut Vec<Diagnostic>,
    in_compensate: bool,
) {
    match &s.node {
        Stmt::Defer {
            kind: DeferKind::Compensate,
            body,
        } => {
            // Everything inside a compensate block is covered.
            for inner in body {
                check_compensation_stmt(inner, tools, diags, true);
            }
        }
        Stmt::Let { init, .. } | Stmt::Var { init, .. } => {
            check_compensation_expr(init, tools, diags, in_compensate);
        }
        Stmt::Expr(e) => check_compensation_expr(e, tools, diags, in_compensate),
        Stmt::Return(e) => {
            if let Some(e) = e {
                check_compensation_expr(e, tools, diags, in_compensate);
            }
        }
        Stmt::If { then, else_, .. } => {
            for s in then {
                check_compensation_stmt(s, tools, diags, in_compensate);
            }
            if let Some(e) = else_ {
                for s in e {
                    check_compensation_stmt(s, tools, diags, in_compensate);
                }
            }
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => {
            for s in body {
                check_compensation_stmt(s, tools, diags, in_compensate);
            }
        }
        Stmt::Match { arms, .. } => {
            for arm in arms {
                for s in &arm.body {
                    check_compensation_stmt(s, tools, diags, in_compensate);
                }
            }
        }
        _ => {}
    }
}

fn check_compensation_expr(
    e: &Spanned<Expr>,
    tools: &[ToolDeclInfo],
    diags: &mut Vec<Diagnostic>,
    in_compensate: bool,
) {
    match &e.node {
        Expr::Call { callee, .. } => {
            if let Expr::Path(path) = &callee.node {
                if path.len() >= 2 {
                    let name = &path.last().unwrap().node;
                    if is_write_name(name) && !in_compensate {
                        // Check if the tool is declared idempotent.
                        let tool_path = path_string(path);
                        let is_idempotent =
                            tools.iter().any(|t| t.path == tool_path && t.idempotent);
                        if !is_idempotent {
                            diags.push(
                                Diagnostic::new(
                                    codes::E_COMPENSATION_MISSING,
                                    Severity::Error,
                                    e.span,
                                    format!(
                                        "Non-idempotent write `{}` in a budgeted task without compensation. \
                                         Add `defer compensate {{ ... }}` or declare the tool `idempotent by ...`.",
                                        tool_path
                                    ),
                                )
                                .with_note("If the task fails after this write, there is no rollback path."),
                            );
                        }
                    }
                }
            }
        }
        Expr::Await(_, body) => {
            // Check branches of parallel/race/quorum.
            match &body.node {
                AwaitBody::All(branches) => {
                    for (_, e) in branches {
                        check_compensation_expr(e, tools, diags, in_compensate);
                    }
                }
                AwaitBody::Map { body, .. } => {
                    for s in body {
                        check_compensation_stmt(s, tools, diags, in_compensate);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
}

// =====================================================================
// Secret taint checking: Secret<T> values cannot flow into model inputs
// =====================================================================

/// Collect the set of variable names that have a Secret<T> type annotation.
fn collect_tainted_vars(block: &Block) -> std::collections::HashSet<String> {
    let mut tainted = std::collections::HashSet::new();
    for stmt in block {
        match &stmt.node {
            Stmt::Let { name, ty, .. } | Stmt::Var { name, ty, .. } => {
                if let Some(t) = ty {
                    if type_contains_secret(&t.node) {
                        tainted.insert(name.node.clone());
                    }
                }
            }
            Stmt::If { then, else_, .. } => {
                tainted.extend(collect_tainted_vars(then));
                if let Some(e) = else_ {
                    tainted.extend(collect_tainted_vars(e));
                }
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                tainted.extend(collect_tainted_vars(body));
            }
            _ => {}
        }
    }
    tainted
}

/// Check if a type references Secret<T>.
fn type_contains_secret(ty: &Ty) -> bool {
    match ty {
        Ty::Named { path, args } => {
            let name = path.last().map(|i| i.node.as_str()).unwrap_or("");
            name == "Secret" && !args.is_empty()
        }
        _ => false,
    }
}

/// Walk expressions and check for secret leaks into model calls.
fn check_taint(
    block: &Block,
    tainted: &std::collections::HashSet<String>,
    diags: &mut Vec<Diagnostic>,
) {
    for stmt in block {
        match &stmt.node {
            Stmt::Let { init, .. } | Stmt::Var { init, .. } => {
                check_taint_expr(init, tainted, diags);
            }
            Stmt::Expr(e) => check_taint_expr(e, tainted, diags),
            Stmt::Return(e) => {
                if let Some(e) = e {
                    check_taint_expr(e, tainted, diags);
                }
            }
            Stmt::If { then, else_, .. } => {
                check_taint(then, tainted, diags);
                if let Some(e) = else_ {
                    check_taint(e, tainted, diags);
                }
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                check_taint(body, tainted, diags);
            }
            Stmt::Match { arms, .. } => {
                for arm in arms {
                    check_taint(&arm.body, tainted, diags);
                }
            }
            Stmt::Check { cond, else_block } => {
                check_taint_expr(cond, tainted, diags);
                if let Some(b) = else_block {
                    check_taint(b, tainted, diags);
                }
            }
            Stmt::Require(e) | Stmt::Ensure(e) => {
                check_taint_expr(e, tainted, diags);
            }
            Stmt::Trace { fields, .. } => {
                for (_, v) in fields {
                    check_taint_expr(v, tainted, diags);
                }
            }
            _ => {}
        }
    }
}

fn check_taint_expr(
    e: &Spanned<Expr>,
    tainted: &std::collections::HashSet<String>,
    diags: &mut Vec<Diagnostic>,
) {
    match &e.node {
        Expr::Infer { spec, .. } => {
            if let Some(input) = &spec.input {
                let leaked = find_tainted_refs(input, tainted);
                if !leaked.is_empty() {
                    diags.push(
                    Diagnostic::new(
                        codes::E_SECRET_LEAK,
                        Severity::Error,
                        e.span,
                        format!(
                            "Secret-tainted variable `{}` flows into a model `infer` input. \
                             Redact it before passing to the model, or use Public<T>.",
                            leaked.join("`, `")
                        ),
                    )
                    .with_note("Use a redact() function to convert Secret<T> to Public<T> before model input."),
                );
                }
            }
            // Also check sub-expressions in the spec
            for c in &spec.constraints {
                check_taint_expr(c, tainted, diags);
            }
        }
        Expr::Call { callee, args } => {
            for a in args {
                check_taint_expr(&a.value, tainted, diags);
            }
            // Don't recurse into callee for taint — only data args matter
            let _ = callee;
        }
        Expr::Bin { lhs, rhs, .. } => {
            check_taint_expr(lhs, tainted, diags);
            check_taint_expr(rhs, tainted, diags);
        }
        Expr::Field { receiver, .. } => check_taint_expr(receiver, tainted, diags),
        Expr::Method { receiver, args, .. } => {
            check_taint_expr(receiver, tainted, diags);
            for a in args {
                check_taint_expr(&a.value, tainted, diags);
            }
        }
        Expr::Await(_, body) => match &body.node {
            AwaitBody::All(branches) => {
                for (_, e) in branches {
                    check_taint_expr(e, tainted, diags);
                }
            }
            _ => {}
        },
        Expr::Record(fields) => {
            for (_, v) in fields {
                check_taint_expr(v, tainted, diags);
            }
        }
        _ => {}
    }
}

/// Find all tainted variable names referenced in an expression.
fn find_tainted_refs(
    e: &Spanned<Expr>,
    tainted: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut found = Vec::new();
    collect_tainted_refs(e, tainted, &mut found);
    found.sort();
    found.dedup();
    found
}

fn collect_tainted_refs(
    e: &Spanned<Expr>,
    tainted: &std::collections::HashSet<String>,
    found: &mut Vec<String>,
) {
    match &e.node {
        Expr::Path(p) => {
            if p.len() == 1 && tainted.contains(&p[0].node) {
                found.push(p[0].node.clone());
            }
        }
        Expr::Field { receiver, .. } => collect_tainted_refs(receiver, tainted, found),
        Expr::Record(fields) => {
            for (_, v) in fields {
                collect_tainted_refs(v, tainted, found);
            }
        }
        Expr::Call { args, .. } => {
            for a in args {
                collect_tainted_refs(&a.value, tainted, found);
            }
        }
        Expr::Bin { lhs, rhs, .. } => {
            collect_tainted_refs(lhs, tainted, found);
            collect_tainted_refs(rhs, tainted, found);
        }
        Expr::Interp(parts) | Expr::Markdown(parts) => {
            for p in parts {
                if let InterpPart::Expr(e) = p {
                    collect_tainted_refs(e, tainted, found);
                }
            }
        }
        _ => {}
    }
}
