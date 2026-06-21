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
}

/// A model's response, before type-directed coercion.
pub struct InferResult {
    pub json: serde_json::Value,
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
