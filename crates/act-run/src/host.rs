//! Host abstraction: the interpreter calls out here for models, tools, state,
//! and traces. Real implementations (HTTP) and a mock for tests live in
//! [`crate::host`]. A `Host` must be `Send + Sync` so `await all` branches can
//! run on separate threads.

use crate::value::Value;

#[derive(Debug)]
pub struct HostError(pub String);

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for HostError {}

pub struct InferRequest<'a> {
    pub goal: Option<&'a Value>,
    pub input: Option<&'a Value>,
    pub constraints: &'a [Value],
    /// A JSON Schema describing the target type, for providers that support
    /// `response_format: { type: "json_schema" }`. `None` on mock/test hosts.
    pub ty_schema: Option<&'a serde_json::Value>,
}

/// A model's response, before type-directed coercion.
pub struct InferResult {
    pub json: serde_json::Value,
    pub confidence: f64,
    pub tokens: u64,
    pub cost: f64,
}

/// A request to verify a prior model output.
pub struct VerifyRequest<'a> {
    /// The original goal/input/constraints (same as the infer request).
    pub goal: Option<&'a Value>,
    pub input: Option<&'a Value>,
    pub constraints: &'a [Value],
    /// The model's output to verify.
    pub output: &'a Value,
}

/// A verifier's verdict.
pub struct VerifyResult {
    /// 0.0–1.0: the verifier's confidence that `output` is correct.
    pub confidence: f64,
    pub tokens: u64,
    pub cost: f64,
}

pub struct ToolResult {
    pub ok: bool,
    pub value: Value,
}

#[derive(Clone)]
pub struct StateCell {
    pub value: Value,
    pub version: i64,
}

/// The boundary between the interpreter and the outside world.
pub trait Host: Send + Sync {
    /// Run a model inference. `model` is the alias/path from `using`.
    fn infer(&self, model: &str, req: InferRequest) -> Result<InferResult, HostError>;

    /// Verify a prior model output with a second model call. Returns a
    /// confidence score (0.0–1.0) that the output is correct. The default
    /// implementation returns the same confidence (no-op), so mock hosts and
    /// hosts without a verifier model fall back gracefully.
    fn verify(&self, model: &str, req: VerifyRequest) -> Result<VerifyResult, HostError> {
        let _ = (model, req);
        Ok(VerifyResult {
            confidence: 1.0,
            tokens: 0,
            cost: 0.0,
        })
    }

    /// Invoke a tool by dotted path (e.g. `gh.create_pull_request`).
    fn call_tool(&self, path: &str, args: Vec<(String, Value)>) -> Result<ToolResult, HostError>;

    /// Read a durable state cell.
    fn state_read(&self, key: &str) -> Result<StateCell, HostError>;

    /// Conditionally write a durable state cell. `expected_version` enforces
    /// optimistic concurrency; the host should reject a stale version.
    fn state_update(
        &self,
        key: &str,
        expected_version: Option<i64>,
        value: Value,
    ) -> Result<StateCell, HostError>;

    /// Record a `trace` checkpoint for later replay.
    fn record_trace(&self, label: &str, fields: Vec<(String, Value)>);

    /// Fetch a previously recorded trace by label (`replay trace("X")`).
    fn replay_trace(&self, label: &str) -> Option<Value>;

    /// Wall-clock milliseconds elapsed since the run began (for budgets).
    fn elapsed_ms(&self) -> u64;
}
