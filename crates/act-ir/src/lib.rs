//! Act IR: lowering from AST to an executable graph IR.
//!
//! The IR is a node graph that a runner can schedule, checkpoint,
//! cancel, and replay. This v1 implements structural lowering of
//! the core constructs: sequential blocks, parallel all/map, calls,
//! infer/decide, and control flow.

use act_diagnostics::{Diagnostic, DiagnosticReport, Severity};
use act_syntax::ast::*;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Graph {
    pub root: NodeId,
    pub nodes: Vec<GraphNode>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub children: Vec<NodeId>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum NodeKind {
    Seq,
    ParallelAll,
    ParallelMap,
    Race,
    Quorum,
    ToolCall,
    ModelCall,
    Decide,
    Let,
    If,
    Match,
    Return,
    Trace,
    Checkpoint,
    Literal,
    Hole,
    Pure,
}

pub struct LowerOutput {
    pub graph: Option<Graph>,
    pub report: DiagnosticReport,
}

pub fn lower(module: &Module) -> LowerOutput {
    let mut diags = Vec::new();
    let mut nodes = Vec::new();
    let mut next_id: u64 = 1;

    let mut root_children = Vec::new();
    for item in &module.items {
        match item {
            Item::Fn(d) | Item::Proc(d) | Item::Task(d) => {
                if let Some(body) = &d.body {
                    let id = lower_block(body, &mut nodes, &mut next_id, &mut diags);
                    root_children.push(id);
                }
            }
            Item::Agent(a) => {
                for h in &a.handlers {
                    let id = lower_block(&h.body, &mut nodes, &mut next_id, &mut diags);
                    root_children.push(id);
                }
            }
            _ => {}
        }
    }

    let root = NodeId(next_id);
    nodes.push(GraphNode {
        id: root,
        kind: NodeKind::Seq,
        children: root_children,
        span: module.span,
    });
    let graph = Graph { root, nodes };
    LowerOutput {
        graph: Some(graph),
        report: DiagnosticReport::new(diags),
    }
}

fn fresh(next: &mut u64) -> NodeId {
    let id = NodeId(*next);
    *next += 1;
    id
}

fn lower_block(
    block: &Block,
    nodes: &mut Vec<GraphNode>,
    next: &mut u64,
    diags: &mut Vec<Diagnostic>,
) -> NodeId {
    let mut children = Vec::new();
    for stmt in block {
        let id = lower_stmt(stmt, nodes, next, diags);
        children.push(id);
    }
    let id = fresh(next);
    // Use span of first/last child if available
    let span = block.first().map(|s| s.span).unwrap_or(Span::dummy());
    nodes.push(GraphNode {
        id,
        kind: NodeKind::Seq,
        children,
        span,
    });
    id
}

fn lower_stmt(
    s: &Spanned<Stmt>,
    nodes: &mut Vec<GraphNode>,
    next: &mut u64,
    diags: &mut Vec<Diagnostic>,
) -> NodeId {
    match &s.node {
        Stmt::Let { init, .. } | Stmt::Var { init, .. } => {
            let child = lower_expr(init, nodes, next, diags);
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Let,
                children: vec![child],
                span: s.span,
            });
            id
        }
        Stmt::Assign { target, value } => {
            let a = lower_expr(target, nodes, next, diags);
            let b = lower_expr(value, nodes, next, diags);
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Let,
                children: vec![a, b],
                span: s.span,
            });
            id
        }
        Stmt::Expr(e) => lower_expr(e, nodes, next, diags),
        Stmt::Return(e) => {
            let child = e.as_ref().map(|e| lower_expr(e, nodes, next, diags));
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Return,
                children: child.into_iter().collect(),
                span: s.span,
            });
            id
        }
        Stmt::If { cond, then, else_ } => {
            let c = lower_expr(cond, nodes, next, diags);
            let t = lower_block(then, nodes, next, diags);
            let e = else_.as_ref().map(|b| lower_block(b, nodes, next, diags));
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::If,
                children: [vec![c, t], e.into_iter().collect()].concat(),
                span: s.span,
            });
            id
        }
        Stmt::Match { scrutinee, arms } => {
            let s_ = lower_expr(scrutinee, nodes, next, diags);
            let mut children = vec![s_];
            for arm in arms {
                children.push(lower_block(&arm.body, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Match,
                children,
                span: s.span,
            });
            id
        }
        Stmt::For { iter, body, .. } => {
            let i = lower_expr(iter, nodes, next, diags);
            let b = lower_block(body, nodes, next, diags);
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::ParallelMap,
                children: vec![i, b],
                span: s.span,
            });
            id
        }
        Stmt::While { .. } => {
            // While loops lower to bounded seq; mark as seq for v1.
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Seq,
                children: vec![],
                span: s.span,
            });
            diags.push(Diagnostic::new(
                "W_WHILE_LOWER",
                Severity::Info,
                s.span,
                "`while` lowered to sequential loop; ensure `max` bound is enforced by runtime.",
            ));
            id
        }
        Stmt::Recover { from, body, .. } => {
            let f = lower_expr(from, nodes, next, diags);
            let b = lower_block(body, nodes, next, diags);
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Seq,
                children: vec![f, b],
                span: s.span,
            });
            id
        }
        Stmt::Defer { body, .. } => lower_block(body, nodes, next, diags),
        Stmt::Require(e) | Stmt::Check(e) | Stmt::Ensure(e) => lower_expr(e, nodes, next, diags),
        Stmt::Trace { fields, .. } => {
            let mut children = Vec::new();
            for (_, v) in fields {
                children.push(lower_expr(v, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Trace,
                children,
                span: s.span,
            });
            id
        }
        Stmt::Checkpoint { body, require, .. } => {
            let b = lower_expr(body, nodes, next, diags);
            let r = require.as_ref().map(|r| lower_expr(r, nodes, next, diags));
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Checkpoint,
                children: [vec![b], r.into_iter().collect()].concat(),
                span: s.span,
            });
            id
        }
        Stmt::Invariant { require, .. } => lower_expr(require, nodes, next, diags),
    }
}

fn lower_expr(
    e: &Spanned<Expr>,
    nodes: &mut Vec<GraphNode>,
    next: &mut u64,
    diags: &mut Vec<Diagnostic>,
) -> NodeId {
    match &e.node {
        Expr::Lit(_) | Expr::Path(_) => {
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Literal,
                children: vec![],
                span: e.span,
            });
            id
        }
        Expr::Call { callee, args } => {
            // tool call?
            let kind = if is_tool_callee(callee) {
                NodeKind::ToolCall
            } else {
                NodeKind::Pure
            };
            let mut children = Vec::new();
            for a in args {
                children.push(lower_expr(&a.value, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind,
                children,
                span: e.span,
            });
            id
        }
        Expr::Method { receiver, args, .. } => {
            let mut children = vec![lower_expr(receiver, nodes, next, diags)];
            for a in args {
                children.push(lower_expr(&a.value, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children,
                span: e.span,
            });
            id
        }
        Expr::Field { receiver, .. } | Expr::Index { receiver, .. } => {
            let id = fresh(next);
            let c = lower_expr(receiver, nodes, next, diags);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children: vec![c],
                span: e.span,
            });
            id
        }
        Expr::Bin { lhs, rhs, .. } => {
            let id = fresh(next);
            let l = lower_expr(lhs, nodes, next, diags);
            let r = lower_expr(rhs, nodes, next, diags);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children: vec![l, r],
                span: e.span,
            });
            id
        }
        Expr::Un { expr, .. } => {
            let id = fresh(next);
            let c = lower_expr(expr, nodes, next, diags);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children: vec![c],
                span: e.span,
            });
            id
        }
        Expr::Try(e) => lower_expr(e, nodes, next, diags),
        Expr::Await(kind, body) => lower_await(*kind, body, nodes, next, diags, e.span),
        Expr::Infer { .. } => {
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::ModelCall,
                children: vec![],
                span: e.span,
            });
            id
        }
        Expr::Decide { source, .. } => {
            let id = fresh(next);
            let c = lower_expr(source, nodes, next, diags);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Decide,
                children: vec![c],
                span: e.span,
            });
            id
        }
        Expr::ResultCtor { value, .. } => {
            let id = fresh(next);
            let children = value
                .as_ref()
                .map(|v| vec![lower_expr(v, nodes, next, diags)])
                .unwrap_or_default();
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children,
                span: e.span,
            });
            id
        }
        Expr::Spawn { .. } => {
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children: vec![],
                span: e.span,
            });
            id
        }
        Expr::Hole(_) => {
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Hole,
                children: vec![],
                span: e.span,
            });
            diags.push(Diagnostic::new(
                act_diagnostics::codes::E_HOLE_UNFILLED,
                Severity::Error,
                e.span,
                "Typed hole is unfilled; cannot execute.",
            ));
            id
        }
        Expr::Record(fields) => {
            let mut children = Vec::new();
            for (_, v) in fields {
                children.push(lower_expr(v, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children,
                span: e.span,
            });
            id
        }
        Expr::Array(elems) => {
            let mut children = Vec::new();
            for e in elems {
                children.push(lower_expr(e, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children,
                span: e.span,
            });
            id
        }
        Expr::Block(b) => lower_block(b, nodes, next, diags),
        Expr::ParallelRecord(fields) => {
            let mut children = Vec::new();
            for (_, v) in fields {
                children.push(lower_expr(v, nodes, next, diags));
            }
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::ParallelAll,
                children,
                span: e.span,
            });
            id
        }
        Expr::Interp(_) | Expr::Markdown(_) => {
            let id = fresh(next);
            nodes.push(GraphNode {
                id,
                kind: NodeKind::Pure,
                children: vec![],
                span: e.span,
            });
            id
        }
    }
}

fn lower_await(
    kind: AwaitKind,
    body: &Spanned<AwaitBody>,
    nodes: &mut Vec<GraphNode>,
    next: &mut u64,
    diags: &mut Vec<Diagnostic>,
    span: Span,
) -> NodeId {
    let (node_kind, children) = match &body.node {
        AwaitBody::All(branches) => {
            let mut c = Vec::new();
            for (_, e) in branches {
                c.push(lower_expr(e, nodes, next, diags));
            }
            (NodeKind::ParallelAll, c)
        }
        AwaitBody::Map { iter, body, .. } => {
            let i = lower_expr(iter, nodes, next, diags);
            let b = lower_block(body, nodes, next, diags);
            (NodeKind::ParallelMap, vec![i, b])
        }
        AwaitBody::Race { branches, .. } => {
            let mut c = Vec::new();
            for (_, e) in branches {
                c.push(lower_expr(e, nodes, next, diags));
            }
            (NodeKind::Race, c)
        }
        AwaitBody::Quorum { branches, .. } => {
            let mut c = Vec::new();
            for (_, e) in branches {
                c.push(lower_expr(e, nodes, next, diags));
            }
            (NodeKind::Quorum, c)
        }
    };
    let _ = kind;
    let id = fresh(next);
    nodes.push(GraphNode {
        id,
        kind: node_kind,
        children,
        span,
    });
    id
}

fn is_tool_callee(callee: &Spanned<Expr>) -> bool {
    if let Expr::Path(p) = &callee.node {
        p.len() >= 2
    } else {
        false
    }
}
