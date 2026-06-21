//! AST interpreter for Act. Executes tasks end-to-end against a [`Host`]:
//! model `infer` calls a real model, tool calls dispatch, `await all` runs
//! branches on separate threads, budgets/capabilities are enforced at runtime,
//! and `trace`/`replay` read and write a real store.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use act_diagnostics::Diagnostic;
use act_syntax::ast::*;

use act_parser::parse_module;

use crate::budget::{BudgetLimits, BudgetTracker};
use crate::host::{Host, HostError, InferRequest};
use crate::registry::{FnRegistry, TypeRegistry};
use crate::value::{coerce, from_literal, Value};

const EMPTY_BLOCK: &Block = &Vec::new();

/// Infrastructure-level failure (not a task's typed `err`).
#[derive(Debug)]
pub enum RunError {
    Host(HostError),
    Budget(Diagnostic),
    Eval(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Host(e) => write!(f, "host error: {}", e),
            RunError::Budget(d) => write!(f, "{}", d.message),
            RunError::Eval(s) => write!(f, "runtime error: {}", s),
        }
    }
}

impl std::error::Error for RunError {}

impl From<HostError> for RunError {
    fn from(e: HostError) -> Self {
        RunError::Host(e)
    }
}

/// Non-local control flow during evaluation.
enum Exn {
    /// `return v`
    Return(Value),
    /// `try` propagating an `err` result value.
    Propagate(Value),
    /// Unrecoverable: host failure or budget exceeded.
    Fatal(RunError),
}

#[derive(Clone, Default)]
struct Env(Vec<HashMap<String, Value>>);

impl Env {
    fn new() -> Env {
        Env(vec![HashMap::new()])
    }
    fn get(&self, name: &str) -> Option<&Value> {
        self.0.iter().rev().find_map(|s| s.get(name))
    }
    fn bind(&mut self, name: String, val: Value) {
        if let Some(scope) = self.0.last_mut() {
            scope.insert(name, val);
        }
    }
    fn push(&mut self) {
        self.0.push(HashMap::new());
    }
    fn pop(&mut self) {
        self.0.pop();
    }
    /// Bind a record's fields into the current scope (for `decide`/`accept`).
    fn bind_record_fields(&mut self, rec: &Value) {
        if let Value::Record(fs) = rec {
            for (k, v) in fs {
                self.bind(k.clone(), v.clone());
            }
        }
    }
}

/// Configuration for a run.
pub struct RunConfig<'h> {
    pub host: &'h dyn Host,
    /// Capability names the runner grants (runtime capability enforcement).
    pub granted_caps: HashSet<String>,
}

/// Run a task by name, returning its (typed Result) value.
pub fn run_task(
    module: &Module,
    task: &str,
    args: Vec<(String, Value)>,
    config: &RunConfig,
) -> Result<Value, RunError> {
    let types = TypeRegistry::from_module(module);
    let fns = FnRegistry::from_module(module);

    let decl = fns
        .get(task)
        .ok_or_else(|| RunError::Eval(format!("no task/fn named `{}` in module", task)))?;

    // Budgets are only meaningful on tasks; fns/procs run unbudgeted.
    let limits = decl.budget.as_ref().map(BudgetLimits::from_budget);
    let budget = BudgetTracker::new(limits.unwrap_or_default());

    let interp = Interp {
        host: config.host,
        budget: &budget,
        types: &types,
        fns: &fns,
        caps: &config.granted_caps,
    };

    let mut env = Env::new();
    bind_params(&mut env, &decl.params, &args);

    match interp.eval_block(&decl.body.clone().unwrap_or_default(), &mut env) {
        Ok(Tail::Some(v)) => Ok(v),
        Ok(Tail::None) => Ok(Value::Null),
        Err(Exn::Return(v)) | Err(Exn::Propagate(v)) => Ok(v),
        Err(Exn::Fatal(e)) => Err(e),
    }
}

/// Run an `eval` block by label. Used to drive `replay trace(...)` assertions.
pub fn run_eval(module: &Module, label: &str, config: &RunConfig) -> Result<Value, RunError> {
    let types = TypeRegistry::from_module(module);
    let fns = FnRegistry::from_module(module);
    let eval = module
        .items
        .iter()
        .find_map(|i| match i {
            Item::Eval(t) if t.label.node == label => Some(t),
            _ => None,
        })
        .ok_or_else(|| RunError::Eval(format!("no eval `{}`", label)))?;
    let budget = BudgetTracker::new(BudgetLimits::default());
    let interp = Interp {
        host: config.host,
        budget: &budget,
        types: &types,
        fns: &fns,
        caps: &config.granted_caps,
    };
    let mut env = Env::new();
    match interp.eval_block(&eval.body, &mut env) {
        Ok(Tail::Some(v)) => Ok(v),
        Ok(Tail::None) => Ok(Value::Null),
        Err(Exn::Return(v)) | Err(Exn::Propagate(v)) => Ok(v),
        Err(Exn::Fatal(e)) => Err(e),
    }
}

fn bind_params(env: &mut Env, params: &[Param], args: &[(String, Value)]) {
    for (i, p) in params.iter().enumerate() {
        let val = args
            .iter()
            .find(|(n, _)| *n == p.name.node)
            .map(|(_, v)| v.clone())
            .or_else(|| args.get(i).map(|(_, v)| v.clone()))
            .or_else(|| {
                p.default.as_ref().and_then(|d| match &d.node {
                    Expr::Lit(l) => Some(from_literal(l)),
                    _ => None,
                })
            })
            .unwrap_or(Value::Null);
        env.bind(p.name.node.clone(), val);
    }
}

/// Serialize the accept-gate inputs into a `VerifyInput` record (each field a
/// JSON string) for the self-hosted `verify` task. JSON strings keep the
/// verifier task fully typed and independent of the caller's types.
fn build_verify_input(
    goal: &Option<Value>,
    input: &Option<Value>,
    constraints: &[Value],
    output: &Value,
) -> Value {
    let json_str = |v: &Value| serde_json::to_string(&crate::value::to_json(v)).unwrap_or_default();
    let goal_str = goal.as_ref().map(json_str).unwrap_or_default();
    let input_str = input.as_ref().map(json_str).unwrap_or_default();
    let constraints_arr: Vec<serde_json::Value> =
        constraints.iter().map(crate::value::to_json).collect();
    let constraints_str =
        serde_json::to_string(&serde_json::Value::Array(constraints_arr)).unwrap_or_default();
    let output_str = json_str(output);
    Value::Record(vec![
        ("goal".into(), Value::String(goal_str)),
        ("input".into(), Value::String(input_str)),
        ("constraints".into(), Value::String(constraints_str)),
        ("output".into(), Value::String(output_str)),
    ])
}

/// Run the self-hosted `verify` task from `builtin/verify.act` against the
/// same host/budget/caps as the caller. Returns the verifier's confidence
/// (0.0–1.0). Host/budget failures propagate as `Fatal`; a typed `err` from
/// the verifier task is also fatal (the verifier could not produce a verdict).
fn run_verify_task(
    host: &dyn Host,
    budget: &BudgetTracker,
    caps: &HashSet<String>,
    verify_input: Value,
) -> Result<f64, Exn> {
    let module = crate::builtin::verify_act();
    let types = TypeRegistry::from_module(module);
    let fns = FnRegistry::from_module(module);
    let decl = fns.get("verify").ok_or_else(|| {
        Exn::Fatal(RunError::Eval(
            "builtin verify task missing from verify.act".into(),
        ))
    })?;
    let interp = Interp {
        host,
        budget,
        types: &types,
        fns: &fns,
        caps,
    };
    let mut env = Env::new();
    bind_params(&mut env, &decl.params, &[("input".into(), verify_input)]);
    let result = match interp.eval_block(&decl.body.clone().unwrap_or_default(), &mut env) {
        Ok(Tail::Some(v)) => v,
        Ok(Tail::None) => Value::Null,
        Err(Exn::Return(v)) => v,
        Err(Exn::Propagate(v)) => {
            return Err(Exn::Fatal(RunError::Eval(format!(
                "verifier task propagated: {:?}",
                v
            ))));
        }
        Err(e @ Exn::Fatal(_)) => return Err(e),
    };
    // The task returns Result<Verdict, String>; unwrap the ok payload.
    let verdict = match result {
        Value::Result {
            ok: true,
            value: Some(boxed),
        } => *boxed,
        _ => {
            return Err(Exn::Fatal(RunError::Eval(
                "verifier task returned err: no verdict".into(),
            )));
        }
    };
    verdict
        .field("confidence")
        .and_then(|c| c.as_f64())
        .ok_or_else(|| {
            Exn::Fatal(RunError::Eval(
                "verifier verdict missing confidence field".into(),
            ))
        })
}

/// Run an arbitrary Act task from source, sharing the caller's host/budget/caps.
/// Used by the `act.run_task` runtime tool so Act programs (e.g. the self-eval
/// harness) can run other Act programs. Traces recorded by the sub-run land in
/// the shared host's store and are replayable by the caller.
fn run_sub_task(
    host: &dyn Host,
    budget: &BudgetTracker,
    caps: &HashSet<String>,
    module_src: &str,
    task: &str,
    args_json: &str,
) -> Result<Value, RunError> {
    let module =
        parse_module(module_src, 0).map_err(|e| RunError::Eval(format!("parse: {:?}", e)))?;
    let types = TypeRegistry::from_module(&module);
    let fns = FnRegistry::from_module(&module);
    let decl = fns
        .get(task)
        .ok_or_else(|| RunError::Eval(format!("no task `{}` in module", task)))?;
    let interp = Interp {
        host,
        budget,
        types: &types,
        fns: &fns,
        caps,
    };
    // Parse the args JSON object into (name, Value) pairs.
    let args: Vec<(String, Value)> = if args_json.trim().is_empty() {
        Vec::new()
    } else {
        let json: serde_json::Value = serde_json::from_str(args_json)
            .map_err(|e| RunError::Eval(format!("args json: {}", e)))?;
        match json {
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| (k, crate::value::value_from_json(&v)))
                .collect(),
            _ => Vec::new(),
        }
    };
    let mut env = Env::new();
    bind_params(&mut env, &decl.params, &args);
    match interp.eval_block(&decl.body.clone().unwrap_or_default(), &mut env) {
        Ok(Tail::Some(v)) => Ok(v),
        Ok(Tail::None) => Ok(Value::Null),
        Err(Exn::Return(v)) | Err(Exn::Propagate(v)) => Ok(v),
        Err(Exn::Fatal(e)) => Err(e),
    }
}

struct Interp<'h> {
    host: &'h dyn Host,
    budget: &'h BudgetTracker,
    types: &'h TypeRegistry,
    fns: &'h FnRegistry,
    caps: &'h HashSet<String>,
}

enum Tail {
    Some(Value),
    None,
}

impl<'h> Interp<'h> {
    fn eval_block(&self, block: &Block, env: &mut Env) -> Result<Tail, Exn> {
        env.push();
        let mut tail = Tail::None;
        let mut defers: Vec<&Block> = Vec::new();
        let mut thrown: Option<Exn> = None;
        for stmt in block {
            // Defer bodies run at block exit; collect them without executing.
            if let Stmt::Defer { body, .. } = &stmt.node {
                defers.push(body);
                continue;
            }
            match self.eval_stmt(stmt, env) {
                Ok(Some(v)) => tail = Tail::Some(v),
                Ok(None) => {}
                Err(e) => {
                    thrown = Some(e);
                    break;
                }
            }
        }
        // Run defers in reverse order (LIFO) on exit, regardless of outcome.
        for body in defers.into_iter().rev() {
            let _ = self.eval_block(body, env);
        }
        env.pop();
        if let Some(e) = thrown {
            return Err(e);
        }
        Ok(tail)
    }

    fn eval_stmt(&self, s: &Spanned<Stmt>, env: &mut Env) -> Result<Option<Value>, Exn> {
        match &s.node {
            Stmt::Let { name, init, .. } | Stmt::Var { name, init, .. } => {
                let v = self.eval_expr(init, env)?;
                env.bind(name.node.clone(), v.clone());
                Ok(None)
            }
            Stmt::Assign { target, value } => {
                let v = self.eval_expr(value, env)?;
                if let Expr::Field { receiver, name } = &target.node {
                    // Best-effort: only support rebinding a record field by name
                    // when the receiver resolves to a bound variable.
                    if let Expr::Path(p) = &receiver.node {
                        if p.len() == 1 {
                            // Mutate in place if the variable holds a record.
                            if let Some(Value::Record(_)) = env.get(&p[0].node) {
                                let mut rec = env.get(&p[0].node).cloned().unwrap();
                                if let Value::Record(fs) = &mut rec {
                                    if let Some(slot) = fs.iter_mut().find(|(n, _)| *n == name.node)
                                    {
                                        slot.1 = v;
                                    }
                                }
                                env.bind(p[0].node.clone(), rec);
                                return Ok(None);
                            }
                        }
                    }
                }
                if let Expr::Path(p) = &target.node {
                    if p.len() == 1 {
                        env.bind(p[0].node.clone(), v);
                        return Ok(None);
                    }
                }
                Err(Exn::Fatal(RunError::Eval(
                    "unsupported assignment target".into(),
                )))
            }
            Stmt::Expr(e) => self.eval_expr(e, env).map(Some),
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => self.eval_expr(e, env)?,
                    None => Value::Null,
                };
                Err(Exn::Return(v))
            }
            Stmt::If {
                cond, then, else_, ..
            } => {
                let c = self.eval_expr(cond, env)?;
                let block = if c.truthy() {
                    then
                } else {
                    else_.as_ref().unwrap_or(EMPTY_BLOCK)
                };
                match self.eval_block(block, env)? {
                    Tail::Some(v) => Ok(Some(v)),
                    Tail::None => Ok(None),
                }
            }
            Stmt::For {
                item, iter, body, ..
            } => {
                let collection = self.eval_expr(iter, env)?;
                if let Value::Array(elems) = collection {
                    for e in elems {
                        env.push();
                        env.bind(item.node.clone(), e);
                        let _ = self.eval_block(body, env)?;
                        env.pop();
                    }
                }
                Ok(None)
            }
            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond, env)?;
                    if !c.truthy() {
                        break;
                    }
                    let _ = self.eval_block(body, env)?;
                }
                Ok(None)
            }
            Stmt::Match { scrutinee, arms } => {
                let v = self.eval_expr(scrutinee, env)?;
                for arm in arms {
                    if let Some(bound) = self.match_pattern(&arm.pattern.node, &v)? {
                        env.push();
                        if let Some(b) = bound {
                            env.bind(self.arm_binder_name(&arm.pattern.node), b);
                        }
                        let outcome = self.eval_block(&arm.body, env)?;
                        env.pop();
                        match outcome {
                            Tail::Some(rv) => return Ok(Some(rv)),
                            Tail::None => return Ok(None),
                        }
                    }
                }
                Ok(None)
            }
            Stmt::Recover { from, body, .. } => match self.eval_expr(from, env) {
                Ok(v) => Ok(Some(v)),
                Err(Exn::Propagate(_)) => match self.eval_block(body, env)? {
                    Tail::Some(v) => Ok(Some(v)),
                    Tail::None => Ok(None),
                },
                Err(other) => Err(other),
            },
            Stmt::Require(e) => {
                let v = self.eval_expr(e, env)?;
                if !v.truthy() {
                    return Err(Exn::Fatal(RunError::Eval(
                        "`require` assertion failed".into(),
                    )));
                }
                Ok(None)
            }
            Stmt::Check {
                cond, else_block, ..
            } => {
                let c = self.eval_expr(cond, env)?;
                if !c.truthy() {
                    if let Some(b) = else_block {
                        return match self.eval_block(b, env)? {
                            Tail::Some(v) => Ok(Some(v)),
                            Tail::None => Ok(None),
                        };
                    }
                    return Err(Exn::Fatal(RunError::Eval(
                        "`check` failed with no else".into(),
                    )));
                }
                Ok(None)
            }
            Stmt::Ensure(e) => {
                let v = self.eval_expr(e, env)?;
                if !v.truthy() {
                    return Err(Exn::Fatal(RunError::Eval("`ensure` failed".into())));
                }
                Ok(None)
            }
            Stmt::Trace { label, fields } => {
                let mut evaled = Vec::with_capacity(fields.len());
                for (n, v) in fields {
                    evaled.push((n.node.clone(), self.eval_expr(v, env)?));
                }
                self.host.record_trace(&label.node, evaled);
                Ok(None)
            }
            Stmt::Checkpoint { .. } | Stmt::Invariant { .. } => Ok(None),
            Stmt::Defer { .. } => Ok(None), // handled by eval_block
        }
    }

    fn match_pattern(&self, pat: &Pattern, v: &Value) -> Result<Option<Option<Value>>, Exn> {
        let bound = match pat {
            Pattern::Wildcard => Some(None),
            Pattern::Bind(_) => Some(Some(v.clone())),
            Pattern::Tag { name, binder } => {
                let tag = name.last().map(|i| i.node.as_str()).unwrap_or("");
                let matches = match v {
                    Value::Result { ok, value: _ } => {
                        (tag == "ok" && *ok) || (tag == "err" && !*ok)
                    }
                    _ => false,
                };
                if matches {
                    Some(binder.as_ref().map(|_| {
                        if let Value::Result {
                            value: Some(inner), ..
                        } = v
                        {
                            (**inner).clone()
                        } else {
                            Value::Null
                        }
                    }))
                } else {
                    None
                }
            }
            Pattern::Lit(e) => {
                let lit = self.eval_expr(e, &Env::new())?;
                if value_eq(&lit, v) {
                    Some(None)
                } else {
                    None
                }
            }
        };
        Ok(bound)
    }

    fn arm_binder_name(&self, pat: &Pattern) -> String {
        match pat {
            Pattern::Tag {
                binder: Some(b), ..
            } => b.node.clone(),
            Pattern::Bind(i) => i.node.clone(),
            _ => "_".to_string(),
        }
    }

    fn eval_expr(&self, e: &Spanned<Expr>, env: &Env) -> Result<Value, Exn> {
        match &e.node {
            Expr::Lit(l) => Ok(from_literal(l)),
            Expr::Path(p) => Ok(self.lookup_path(p, env)),
            Expr::Interp(parts) | Expr::Markdown(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        InterpPart::Str(s) => out.push_str(s),
                        InterpPart::Expr(x) => match self.eval_expr(x, env)? {
                            Value::String(s) => out.push_str(&s),
                            other => out.push_str(&format!("{:?}", other)),
                        },
                    }
                }
                Ok(Value::String(out))
            }
            Expr::Call { callee, args } => self.eval_call(callee, args, env),
            Expr::Method {
                receiver,
                name,
                args,
            } => self.eval_method(receiver, name, args, env),
            Expr::Field { receiver, name } => {
                let r = self.eval_expr(receiver, env)?;
                Ok(r.field(&name.node).cloned().unwrap_or(Value::Null))
            }
            Expr::Index { receiver, index } => {
                let r = self.eval_expr(receiver, env)?;
                let i = self.eval_expr(index, env)?;
                match (&r, &i) {
                    (Value::Array(a), Value::Int(n)) => {
                        Ok(a.get(*n as usize).cloned().unwrap_or(Value::Null))
                    }
                    _ => Ok(Value::Null),
                }
            }
            Expr::Bin { op, lhs, rhs } => {
                let l = self.eval_expr(lhs, env)?;
                // Short-circuit && / ||
                if *op == BinOp::And {
                    return if l.truthy() {
                        self.eval_expr(rhs, env)
                    } else {
                        Ok(l)
                    };
                }
                if *op == BinOp::Or {
                    return if l.truthy() {
                        Ok(l)
                    } else {
                        self.eval_expr(rhs, env)
                    };
                }
                let r = self.eval_expr(rhs, env)?;
                Ok(eval_binop(*op, &l, &r))
            }
            Expr::Un { op, expr } => {
                let v = self.eval_expr(expr, env)?;
                Ok(match (op, v) {
                    (UnOp::Neg, Value::Int(n)) => Value::Int(-n),
                    (UnOp::Neg, Value::Decimal(d)) => Value::Decimal(-d),
                    (UnOp::Not, Value::Bool(b)) => Value::Bool(!b),
                    _ => Value::Null,
                })
            }
            Expr::Try(inner) => {
                let v = self.eval_expr(inner, env)?;
                match v {
                    Value::Result { ok: true, value } => {
                        Ok(value.map(|b| *b).unwrap_or(Value::Null))
                    }
                    err @ Value::Result { ok: false, .. } => Err(Exn::Propagate(err)),
                    other => Ok(other),
                }
            }
            Expr::ResultCtor { variant, value } => {
                let inner = match value {
                    Some(v) => Some(Box::new(self.eval_expr(v, env)?)),
                    None => None,
                };
                Ok(Value::Result {
                    ok: *variant == ResultVariant::Ok,
                    value: inner,
                })
            }
            Expr::Record(fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (n, v) in fields {
                    out.push((n.node.clone(), self.eval_expr(v, env)?));
                }
                Ok(Value::Record(out))
            }
            Expr::Array(elems) => {
                let mut out = Vec::with_capacity(elems.len());
                for e in elems {
                    out.push(self.eval_expr(e, env)?);
                }
                Ok(Value::Array(out))
            }
            Expr::Block(b) => {
                let mut env = env.clone();
                match self.eval_block(b, &mut env)? {
                    Tail::Some(v) => Ok(v),
                    Tail::None => Ok(Value::Null),
                }
            }
            Expr::ParallelRecord(fields) => {
                let mut out = Vec::with_capacity(fields.len());
                for (n, v) in fields {
                    out.push((n.node.clone(), self.eval_expr(v, env)?));
                }
                Ok(Value::Record(out))
            }
            Expr::Await(_, body) => self.eval_await(&body.node, env),
            Expr::Infer { ty, model, spec } => self.eval_infer(ty, model, spec, env),
            Expr::Decide {
                ty: _,
                source,
                score_by,
                require,
                else_,
            } => self.eval_decide(source, score_by, require.as_deref(), else_.as_ref(), env),
            Expr::Spawn { .. } => Err(Exn::Fatal(RunError::Eval(
                "spawn is not supported by the kernel runtime".into(),
            ))),
            Expr::Hole(_) => Err(Exn::Fatal(RunError::Eval(
                "cannot execute an unfilled typed hole `??`".into(),
            ))),
            Expr::Replay { label } => {
                let lv = self.eval_expr(label, env)?;
                if let Value::String(s) = lv {
                    self.host
                        .replay_trace(&s)
                        .ok_or_else(|| Exn::Fatal(RunError::Eval(format!("no trace `{}`", s))))
                } else {
                    Err(Exn::Fatal(RunError::Eval(
                        "replay label must be a string".into(),
                    )))
                }
            }
        }
    }

    fn lookup_path(&self, p: &[Ident], env: &Env) -> Value {
        // A dotted path in value position is field access on a bound variable:
        // `best.confidence`, `results.logs`, `best.hypothesis.claim`.
        if p.is_empty() {
            return Value::Null;
        }
        let mut v = match env.get(&p[0].node) {
            Some(v) => v.clone(),
            None => {
                return match p.len() {
                    1 => match p[0].node.as_str() {
                        "true" => Value::Bool(true),
                        "false" => Value::Bool(false),
                        "null" => Value::Null,
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                };
            }
        };
        for seg in &p[1..] {
            v = v.field(&seg.node).cloned().unwrap_or(Value::Null);
        }
        v
    }

    fn eval_call(&self, callee: &Spanned<Expr>, args: &[CallArg], env: &Env) -> Result<Value, Exn> {
        let path = match &callee.node {
            Expr::Path(p) => p,
            _ => {
                let v = self.eval_expr(callee, env)?;
                return Ok(v);
            }
        };

        // State cells.
        if path.len() == 2 && path[0].node == "state" {
            return self.eval_state(&path[1].node, args, env);
        }
        // Builtins.
        if path.len() == 1 {
            match path[0].node.as_str() {
                "redact" => {
                    let v = self.eval_arg(args, 0, env)?;
                    return Ok(match v {
                        Value::Secret(inner) => *inner,
                        other => other,
                    });
                }
                "len" => {
                    let v = self.eval_arg(args, 0, env)?;
                    return Ok(match v {
                        Value::Array(a) => Value::Int(a.len() as i64),
                        Value::String(s) => Value::Int(s.len() as i64),
                        _ => Value::Null,
                    });
                }
                _ => {}
            }
            // User-defined fn/proc/task.
            if self.fns.get(&path[0].node).is_some() {
                let evaled = self.eval_args(args, env)?;
                return self.call_fn(&path[0].node, evaled);
            }
        }
        // Tool call: dotted path like gh.create_pull_request.
        if path.len() >= 2 {
            let full = path_string(path);
            // Capability enforcement.
            self.enforce_cap(&full)?;
            self.budget
                .spend_tool_call()
                .map_err(|d| Exn::Fatal(RunError::Budget(d)))?;
            let evaled = self.eval_args(args, env)?;

            // Runtime tools (`act.*`) are self-hosted: the interp handles them
            // directly because they need interp recursion (a host can't borrow
            // itself to call back in). `act.run_task` runs another Act program
            // from source, sharing this run's host/budget/caps; traces from the
            // sub-run land in the shared host store and are replayable here.
            if let Some(rest) = full.strip_prefix("act.") {
                return self.runtime_tool(rest, &evaled);
            }

            let res = self
                .host
                .call_tool(&full, evaled)
                .map_err(|e| Exn::Fatal(RunError::Host(e)))?;
            Ok(Value::Result {
                ok: res.ok,
                value: Some(Box::new(res.value)),
            })
        } else {
            Ok(Value::Null)
        }
    }

    fn eval_method(
        &self,
        receiver: &Spanned<Expr>,
        name: &Ident,
        args: &[CallArg],
        env: &Env,
    ) -> Result<Value, Exn> {
        let r = self.eval_expr(receiver, env)?;
        match name.node.as_str() {
            "len" => Ok(match r {
                Value::Array(a) => Value::Int(a.len() as i64),
                Value::String(s) => Value::Int(s.len() as i64),
                _ => Value::Null,
            }),
            other => {
                let _ = (args, other);
                Err(Exn::Fatal(RunError::Eval(format!(
                    "unknown method `{}`",
                    name.node
                ))))
            }
        }
    }

    fn eval_state(&self, op: &str, args: &[CallArg], env: &Env) -> Result<Value, Exn> {
        let evaled = self.eval_args(args, env)?;
        let key = evaled
            .iter()
            .find(|(n, _)| n == "key")
            .map(|(_, v)| v.clone())
            .or_else(|| evaled.first().map(|(_, v)| v.clone()))
            .unwrap_or(Value::Null);
        let key = match key {
            Value::String(s) => s,
            _ => {
                return Err(Exn::Fatal(RunError::Eval(
                    "state key must be a string".into(),
                )))
            }
        };
        match op {
            "read" => {
                let cell = self
                    .host
                    .state_read(&key)
                    .map_err(|e| Exn::Fatal(RunError::Host(e)))?;
                Ok(state_cell_value(cell))
            }
            "update" => {
                let expected = evaled
                    .iter()
                    .find(|(n, _)| n == "expected_version")
                    .and_then(|(_, v)| match v {
                        Value::Int(n) => Some(*n),
                        _ => None,
                    });
                let value = evaled
                    .iter()
                    .find(|(n, _)| n == "value")
                    .map(|(_, v)| v.clone())
                    .unwrap_or(Value::Null);
                let cell = self
                    .host
                    .state_update(&key, expected, value)
                    .map_err(|e| Exn::Fatal(RunError::Host(e)))?;
                Ok(state_cell_value(cell))
            }
            _ => Err(Exn::Fatal(RunError::Eval(format!(
                "unknown state op `{}`",
                op
            )))),
        }
    }

    fn call_fn(&self, name: &str, args: Vec<(String, Value)>) -> Result<Value, Exn> {
        let decl = self.fns.get(name).unwrap();
        let mut env = Env::new();
        bind_params(&mut env, &decl.params, &args);
        let body = decl.body.clone().unwrap_or_default();
        match self.eval_block(&body, &mut env) {
            Ok(Tail::Some(v)) => Ok(v),
            Ok(Tail::None) => Ok(Value::Null),
            Err(Exn::Return(v)) | Err(Exn::Propagate(v)) => Ok(v),
            Err(Exn::Fatal(e)) => Err(Exn::Fatal(e)),
        }
    }

    fn eval_arg(&self, args: &[CallArg], idx: usize, env: &Env) -> Result<Value, Exn> {
        match args.get(idx) {
            Some(a) => self.eval_expr(&a.value, env),
            None => Ok(Value::Null),
        }
    }

    fn eval_args(&self, args: &[CallArg], env: &Env) -> Result<Vec<(String, Value)>, Exn> {
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let name = a
                .name
                .as_ref()
                .map(|n| n.node.clone())
                .unwrap_or_else(|| format!("_{}", i));
            out.push((name, self.eval_expr(&a.value, env)?));
        }
        Ok(out)
    }

    fn enforce_cap(&self, tool_path: &str) -> Result<(), Exn> {
        // If the runner granted a non-empty cap set, require the tool prefix
        // to be present. An empty cap set means "unrestricted" (tests/grants-all).
        if self.caps.is_empty() {
            return Ok(());
        }
        let prefix = tool_path.split('.').next().unwrap_or(tool_path);
        if self.caps.iter().any(|c| c == prefix || c == tool_path) {
            Ok(())
        } else {
            Err(Exn::Fatal(RunError::Eval(format!(
                "capability for `{}` not granted at runtime",
                tool_path
            ))))
        }
    }

    /// Dispatch a self-hosted `act.*` runtime tool. These are handled in the
    /// interp (not the host) because they require interp recursion.
    fn runtime_tool(&self, op: &str, args: &[(String, Value)]) -> Result<Value, Exn> {
        let text = |name: &str| -> Option<String> {
            args.iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, v)| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
        };
        match op {
            // act.run_task(module_src: String, task: String, args: String) ->
            //   Result<String, String>. Runs another Act program from source
            //   against the shared host/budget/caps. The result is JSON-
            //   stringified so the caller can pass it to `infer` for scoring;
            //   traces from the sub-run are in the shared host store.
            "run_task" => {
                let module_src = text("module_src").unwrap_or_default();
                let task = text("task").unwrap_or_default();
                let args_json = text("args").unwrap_or_default();
                match run_sub_task(
                    self.host,
                    self.budget,
                    self.caps,
                    &module_src,
                    &task,
                    &args_json,
                ) {
                    Ok(v) => Ok(Value::Result {
                        ok: true,
                        value: Some(Box::new(Value::String(
                            serde_json::to_string(&crate::value::to_json(&v))
                                .unwrap_or_else(|_| "null".to_string()),
                        ))),
                    }),
                    Err(e) => Ok(Value::Result {
                        ok: false,
                        value: Some(Box::new(Value::String(format!("{}", e)))),
                    }),
                }
            }
            other => Err(Exn::Fatal(RunError::Eval(format!(
                "unknown runtime tool `act.{}`",
                other
            )))),
        }
    }

    fn eval_await(&self, body: &AwaitBody, env: &Env) -> Result<Value, Exn> {
        match body {
            AwaitBody::All(branches) => self.parallel_all(branches, env).map(Value::Record),
            AwaitBody::Map {
                item,
                iter,
                parallel,
                body,
                ..
            } => {
                let collection = self.eval_expr(iter, env)?;
                let elems = match collection {
                    Value::Array(a) => a,
                    _ => vec![],
                };
                let _ = parallel; // run sequentially; parallelism here is optional
                let mut out = Vec::with_capacity(elems.len());
                for e in elems {
                    let mut env = env.clone();
                    env.push();
                    env.bind(item.node.clone(), e);
                    match self.eval_block(body, &mut env)? {
                        Tail::Some(v) => out.push(v),
                        Tail::None => out.push(Value::Null),
                    }
                }
                Ok(Value::Array(out))
            }
            AwaitBody::Race { branches, .. } | AwaitBody::Quorum { branches, .. } => {
                // Run all, return the first ok result payload.
                let record = self.parallel_all(branches, env)?;
                for (_, v) in &record {
                    if let Value::Result { ok: true, value } = v {
                        return Ok(value.clone().map(|b| *b).unwrap_or(Value::Null));
                    }
                }
                Ok(record
                    .into_iter()
                    .next()
                    .map(|(_, v)| v)
                    .unwrap_or(Value::Null))
            }
        }
    }

    /// Evaluate named branches in parallel on separate threads, joining into a
    /// record. The first thrown control flow propagates.
    fn parallel_all(
        &self,
        branches: &[(Ident, Spanned<Expr>)],
        env: &Env,
    ) -> Result<Vec<(String, Value)>, Exn> {
        let shared = Arc::new(());
        let results = std::thread::scope(|s| {
            let handles: Vec<_> = branches
                .iter()
                .map(|(name, expr)| {
                    let env = env.clone();
                    let shared = shared.clone();
                    let _ = shared;
                    s.spawn(move || (name.node.clone(), self.eval_expr(expr, &env)))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("await branch panicked"))
                .collect::<Vec<_>>()
        });
        let mut out = Vec::with_capacity(results.len());
        for (name, res) in results {
            out.push((name, res?));
        }
        Ok(out)
    }

    fn eval_infer(
        &self,
        ty: &Spanned<Ty>,
        model: &Spanned<Expr>,
        spec: &InferSpec,
        env: &Env,
    ) -> Result<Value, Exn> {
        let model_name = path_of(&model.node);
        let goal = self.maybe_eval(spec.goal.as_deref(), env)?;
        let input = self.maybe_eval(spec.input.as_deref(), env)?;
        let mut constraints = Vec::new();
        for c in &spec.constraints {
            constraints.push(self.eval_expr(c, env)?);
        }
        let schema = crate::schema::ty_to_schema(&ty.node, self.types);
        let req = InferRequest {
            goal: goal.as_ref(),
            input: input.as_ref(),
            constraints: &constraints,
            ty_schema: Some(&schema),
        };
        let res = self
            .host
            .infer(&model_name, req)
            .map_err(|e| Exn::Fatal(RunError::Host(e)))?;
        self.budget
            .spend_model(res.tokens, res.cost)
            .map_err(|d| Exn::Fatal(RunError::Budget(d)))?;
        let value = coerce(&ty.node, &res.json, self.types).map_err(|e| {
            Exn::Fatal(RunError::Eval(format!(
                "model output coercion failed: {}",
                e
            )))
        })?;
        // Accept gate: bind `confidence` and the result fields, then evaluate.
        if let Some(accept) = &spec.accept {
            // Self-hosted verifier: dispatch through the `verify` task in
            // `builtin/verify.act` instead of a host primitive. Verification
            // is then subject to the same budgets, effects, and traces as user
            // code — the difference between "model was fluent" and "model was
            // right", made auditable through the language's own `trace`/`replay`.
            // The verifier's `infer` has no `accept` clause, so this cannot
            // recurse. Any host/budget failure propagates as fatal (matching the
            // previous host-`verify` semantics); mock hosts return a canned
            // verdict (confidence 1.0), so the gate only bites under a real
            // verifier model.
            let verify_input = build_verify_input(&goal, &input, &constraints, &value);
            let verified = run_verify_task(self.host, self.budget, self.caps, verify_input)?;
            let effective_confidence = verified;
            let mut env2 = env.clone();
            env2.push();
            env2.bind(
                "confidence".to_string(),
                Value::Decimal(effective_confidence),
            );
            env2.bind_record_fields(&value);
            let passed = self.eval_expr(accept, &env2)?.truthy();
            if !passed {
                if let Some(els) = &spec.else_ {
                    let mut env3 = env.clone();
                    return match self.eval_block(els, &mut env3)? {
                        Tail::Some(v) => Err(Exn::Return(v)),
                        Tail::None => Err(Exn::Return(Value::Null)),
                    };
                }
                return Err(Exn::Fatal(RunError::Eval(
                    "infer accept gate failed with no else".into(),
                )));
            }
        }
        Ok(value)
    }

    fn eval_decide(
        &self,
        source: &Spanned<Expr>,
        score_by: &[ScoreClause],
        require: Option<&Spanned<Expr>>,
        else_: Option<&Block>,
        env: &Env,
    ) -> Result<Value, Exn> {
        let src = self.eval_expr(source, env)?;
        let candidates = match src {
            Value::Array(a) => a,
            other => vec![other],
        };
        // Score each candidate by weighted sum of named fields (desc positive).
        let mut scored: Vec<(f64, Value)> = candidates
            .into_iter()
            .map(|c| (self.score_candidate(&c, score_by), c))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, c) in scored {
            if let Some(req) = require {
                let mut env2 = env.clone();
                env2.push();
                env2.bind_record_fields(&c);
                if !self.eval_expr(req, &env2)?.truthy() {
                    continue;
                }
            }
            return Ok(c);
        }
        // Nothing satisfied: evaluate else.
        if let Some(els) = else_ {
            let mut env2 = env.clone();
            return match self.eval_block(els, &mut env2)? {
                Tail::Some(v) => Ok(v),
                Tail::None => Ok(Value::Null),
            };
        }
        Ok(Value::Null)
    }

    fn score_candidate(&self, c: &Value, score_by: &[ScoreClause]) -> f64 {
        let mut score = 0.0;
        for clause in score_by {
            let weight = clause
                .weight
                .as_ref()
                .and_then(|w| {
                    if let Expr::Lit(Literal::Decimal(s)) = &w.node {
                        s.parse::<f64>().ok()
                    } else if let Expr::Lit(Literal::Int(n)) = &w.node {
                        Some(*n as f64)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);
            let field = clause.field.last().map(|i| i.node.as_str()).unwrap_or("");
            let val = c.field(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
            score += match clause.dir {
                SortDir::Desc => weight * val,
                SortDir::Asc => -weight * val,
            };
        }
        score
    }

    fn maybe_eval(&self, e: Option<&Spanned<Expr>>, env: &Env) -> Result<Option<Value>, Exn> {
        match e {
            Some(e) => self.eval_expr(e, env).map(Some),
            None => Ok(None),
        }
    }
}

fn state_cell_value(cell: crate::host::StateCell) -> Value {
    Value::Record(vec![
        ("value".to_string(), cell.value),
        ("version".to_string(), Value::Int(cell.version)),
    ])
}

fn path_of(e: &Expr) -> String {
    if let Expr::Path(p) = e {
        path_string(p)
    } else {
        String::new()
    }
}

fn path_string(p: &[Ident]) -> String {
    p.iter()
        .map(|i| i.node.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

fn eval_binop(op: BinOp, l: &Value, r: &Value) -> Value {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            let a = l.as_f64().unwrap_or(0.0);
            let b = r.as_f64().unwrap_or(0.0);
            let v = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => a % b,
                _ => 0.0,
            };
            if matches!(l, Value::Int(_)) && matches!(r, Value::Int(_)) {
                Value::Int(v as i64)
            } else {
                Value::Decimal(v)
            }
        }
        BinOp::Eq => Value::Bool(value_eq(l, r)),
        BinOp::Ne => Value::Bool(!value_eq(l, r)),
        BinOp::Lt => Value::Bool(cmp_num(l, r) == std::cmp::Ordering::Less),
        BinOp::Le => Value::Bool(cmp_num(l, r) != std::cmp::Ordering::Greater),
        BinOp::Gt => Value::Bool(cmp_num(l, r) == std::cmp::Ordering::Greater),
        BinOp::Ge => Value::Bool(cmp_num(l, r) != std::cmp::Ordering::Less),
        BinOp::In => Value::Bool(match r {
            Value::Array(a) => a.iter().any(|x| value_eq(x, l)),
            _ => false,
        }),
        BinOp::NotIn => Value::Bool(!match r {
            Value::Array(a) => a.iter().any(|x| value_eq(x, l)),
            _ => false,
        }),
        BinOp::And | BinOp::Or | BinOp::Pipe => Value::Null,
    }
}

fn cmp_num(l: &Value, r: &Value) -> std::cmp::Ordering {
    let a = l.as_f64().unwrap_or(0.0);
    let b = r.as_f64().unwrap_or(0.0);
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

fn value_eq(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Decimal(a), Value::Decimal(b)) => a == b,
        (Value::Int(a), Value::Decimal(b)) | (Value::Decimal(b), Value::Int(a)) => {
            (*a as f64) == *b
        }
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => false,
    }
}
