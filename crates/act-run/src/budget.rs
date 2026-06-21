//! Runtime budget enforcement. Limits come from a task's `budget` declaration;
//! spend is tracked with atomics so parallel `await all` branches share one
//! counter. Exceeding any limit aborts the run with a typed error.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use act_diagnostics::{codes, Diagnostic, Severity};
use act_syntax::ast::{Budget, BudgetMetric, BudgetOp, Span};

/// Per-metric budget limits parsed from a `budget { ... }` declaration.
/// `None` means unbounded (the metric was not declared).
#[derive(Default, Clone, Debug)]
pub struct BudgetLimits {
    pub wall_ms: Option<u64>,
    pub tokens: Option<u64>,
    pub cost: Option<u64>, // micros
    pub tool_calls: Option<u64>,
}

impl BudgetLimits {
    pub fn from_budget(b: &Budget) -> BudgetLimits {
        let mut lim = BudgetLimits::default();
        for limit in &b.limits {
            // Only <= limits map cleanly to a ceiling; others are ignored.
            if limit.op != BudgetOp::Le {
                continue;
            }
            let val = expr_as_u64(&limit.value.node);
            match limit.metric {
                BudgetMetric::WallTime => lim.wall_ms = val.map(secs_to_ms),
                BudgetMetric::Tokens => lim.tokens = val,
                BudgetMetric::Cost => lim.cost = val.map(|v| v * 1_000_000),
                BudgetMetric::ToolCalls => lim.tool_calls = val,
            }
        }
        lim
    }
}

fn secs_to_ms(v: u64) -> u64 {
    // Duration literals like `30m` are currently lexed as opaque; if a numeric
    // bound is used directly, treat it as milliseconds.
    v
}

fn expr_as_u64(_e: &act_syntax::ast::Expr) -> Option<u64> {
    // Budget bounds are expression nodes; the common case is an Int literal.
    if let act_syntax::ast::Expr::Lit(act_syntax::ast::Literal::Int(n)) = _e {
        return Some(*n as u64);
    }
    None
}

pub struct BudgetTracker {
    start: Instant,
    limits: BudgetLimits,
    tokens: AtomicU64,
    cost_micros: AtomicU64,
    tool_calls: AtomicU64,
}

impl BudgetTracker {
    pub fn new(limits: BudgetLimits) -> BudgetTracker {
        BudgetTracker {
            start: Instant::now(),
            limits,
            tokens: AtomicU64::new(0),
            cost_micros: AtomicU64::new(0),
            tool_calls: AtomicU64::new(0),
        }
    }

    pub fn spend_model(&self, tokens: u64, cost: f64) -> Result<(), Diagnostic> {
        self.tokens.fetch_add(tokens, Ordering::Relaxed);
        self.cost_micros
            .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
        self.assert_ok()
    }

    pub fn spend_tool_call(&self) -> Result<(), Diagnostic> {
        self.tool_calls.fetch_add(1, Ordering::Relaxed);
        self.assert_ok()
    }

    pub fn assert_ok(&self) -> Result<(), Diagnostic> {
        let elapsed = self.start.elapsed().as_millis() as u64;
        if let Some(limit) = self.limits.wall_ms {
            if elapsed > limit {
                return Err(budget_err(elapsed, "wall_time", limit));
            }
        }
        if let Some(limit) = self.limits.tokens {
            let spent = self.tokens.load(Ordering::Relaxed);
            if spent > limit {
                return Err(budget_err(spent, "tokens", limit));
            }
        }
        if let Some(limit) = self.limits.cost {
            let spent = self.cost_micros.load(Ordering::Relaxed);
            if spent > limit {
                return Err(budget_err(spent, "cost", limit));
            }
        }
        if let Some(limit) = self.limits.tool_calls {
            let spent = self.tool_calls.load(Ordering::Relaxed);
            if spent > limit {
                return Err(budget_err(spent, "tool_calls", limit));
            }
        }
        Ok(())
    }
}

fn budget_err(spent: u64, metric: &str, limit: u64) -> Diagnostic {
    let span = Span::dummy();
    Diagnostic::new(
        codes::E_BUDGET_EXCEEDED,
        Severity::Error,
        span,
        format!(
            "Budget exceeded: {} spent {} exceeds limit {}.",
            metric, spent, limit
        ),
    )
}
