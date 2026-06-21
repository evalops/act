//! Act diagnostics: structured, JSON-serializable, repair-oriented.

use act_syntax::ast::Span;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: Severity,
    pub span: SpanSer,
    pub message: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub suggested_patch: Option<SuggestedPatch>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpanSer {
    pub file: u32,
    pub start: u32,
    pub end: u32,
}

impl From<Span> for SpanSer {
    fn from(s: Span) -> SpanSer {
        SpanSer {
            file: s.file,
            start: s.start,
            end: s.end,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SuggestedPatch {
    pub replace: String,
    pub with: String,
}

impl Diagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: Severity,
        span: Span,
        message: impl Into<String>,
    ) -> Diagnostic {
        Diagnostic {
            code: code.into(),
            severity,
            span: span.into(),
            message: message.into(),
            expected: None,
            actual: None,
            suggested_patch: None,
            notes: Vec::new(),
        }
    }
    pub fn with_expected(mut self, e: impl Into<String>) -> Self {
        self.expected = Some(e.into());
        self
    }
    pub fn with_actual(mut self, a: impl Into<String>) -> Self {
        self.actual = Some(a.into());
        self
    }
    pub fn with_patch(mut self, replace: impl Into<String>, with: impl Into<String>) -> Self {
        self.suggested_patch = Some(SuggestedPatch {
            replace: replace.into(),
            with: with.into(),
        });
        self
    }
    pub fn with_note(mut self, n: impl Into<String>) -> Self {
        self.notes.push(n.into());
        self
    }
}

/// Standard diagnostic codes.
pub mod codes {
    pub const E_TYPE_MISMATCH: &str = "E_TYPE_MISMATCH";
    pub const E_EFFECT_MISSING: &str = "E_EFFECT_MISSING";
    pub const E_CAPABILITY_MISSING: &str = "E_CAPABILITY_MISSING";
    pub const E_UNBOUNDED_LOOP: &str = "E_UNBOUNDED_LOOP";
    pub const E_SECRET_LEAK: &str = "E_SECRET_LEAK";
    pub const E_UNHANDLED_RESULT: &str = "E_UNHANDLED_RESULT";
    pub const E_POLICY_CONFLICT: &str = "E_POLICY_CONFLICT";
    pub const E_BUDGET_MISSING: &str = "E_BUDGET_MISSING";
    pub const E_BUDGET_EXCEEDED: &str = "E_BUDGET_EXCEEDED";
    pub const E_HOLE_UNFILLED: &str = "E_HOLE_UNFILLED";
    pub const E_NONDETERMINISTIC_PURE_FN: &str = "E_NONDETERMINISTIC_PURE_FN";
    pub const E_TOOL_RETURN_UNVALIDATED: &str = "E_TOOL_RETURN_UNVALIDATED";
    pub const E_LOW_CONFIDENCE_UNHANDLED: &str = "E_LOW_CONFIDENCE_UNHANDLED";
    pub const E_CHECK_UNHANDLED: &str = "E_CHECK_UNHANDLED";
    pub const E_COMPENSATION_MISSING: &str = "E_COMPENSATION_MISSING";
    pub const E_POLICY_MAY_UNGRANTED: &str = "E_POLICY_MAY_UNGRANTED";
    pub const E_POLICY_MUST_NOT_GRANTED: &str = "E_POLICY_MUST_NOT_GRANTED";
    pub const E_CONFIDENCE_PROVENANCE_MISMATCH: &str = "E_CONFIDENCE_PROVENANCE_MISMATCH";
    pub const E_STATE_UPDATE_UNGUARDED: &str = "E_STATE_UPDATE_UNGUARDED";
    pub const E_REPLAY_WITHOUT_TRACE: &str = "E_REPLAY_WITHOUT_TRACE";
    pub const W_MODEL_CONFIDENCE_HIGH_THRESHOLD: &str = "W_MODEL_CONFIDENCE_HIGH_THRESHOLD";
    pub const E_PARSE: &str = "E_PARSE";
    pub const E_LEX: &str = "E_LEX";
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticReport {
    pub ok: bool,
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticReport {
    pub fn new(diagnostics: Vec<Diagnostic>) -> DiagnosticReport {
        let ok = diagnostics.iter().all(|d| d.severity != Severity::Error);
        DiagnosticReport { ok, diagnostics }
    }
}
