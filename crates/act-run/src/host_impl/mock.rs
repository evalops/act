//! A mock [`Host`] for deterministic tests. Tools and models return canned
//! responses keyed by path; state and traces are held in interior-mutable
//! stores so the host stays `Send + Sync` for parallel `await all` branches.

use std::sync::{Mutex, OnceLock};

use crate::host::{Host, HostError, InferRequest, InferResult, StateCell, ToolResult};
use crate::value::{to_json, Value};

/// Canned model response builder.
pub struct MockInfer {
    pub json: serde_json::Value,
    pub confidence: f64,
    pub tokens: u64,
    pub cost: f64,
}

/// Canned tool response builder.
pub struct MockTool {
    pub ok: bool,
    pub value: Value,
}

/// A mock host. Configure it with [`MockHost::tool`] / [`MockHost::model`] for
/// deterministic outputs; unconfigured calls fall back to sensible defaults.
pub struct MockHost {
    tools: Mutex<Vec<(String, MockTool)>>,
    models: Mutex<Vec<(String, MockInfer)>>,
    state: Mutex<Vec<(String, StateCell)>>,
    traces: Mutex<Vec<(String, Value)>>,
    start: std::time::Instant,
}

impl Default for MockHost {
    fn default() -> Self {
        Self::new()
    }
}

impl MockHost {
    pub fn new() -> MockHost {
        MockHost {
            tools: Mutex::new(Vec::new()),
            models: Mutex::new(Vec::new()),
            state: Mutex::new(Vec::new()),
            traces: Mutex::new(Vec::new()),
            start: std::time::Instant::now(),
        }
    }

    /// Register a canned tool response. `path` is matched by suffix (e.g.
    /// `create_pull_request` matches `gh.create_pull_request`).
    pub fn tool(self, path: &str, ok: bool, value: Value) -> Self {
        self.tools
            .lock()
            .unwrap()
            .push((path.to_string(), MockTool { ok, value }));
        self
    }

    /// Register a canned model response, matched by model alias.
    pub fn model(self, alias: &str, json: serde_json::Value, confidence: f64) -> Self {
        self.models.lock().unwrap().push((
            alias.to_string(),
            MockInfer {
                json,
                confidence,
                tokens: 0,
                cost: 0.0,
            },
        ));
        self
    }

    /// Seed a state cell so `state.read` returns it.
    pub fn seed_state(self, key: &str, value: Value, version: i64) -> Self {
        self.state
            .lock()
            .unwrap()
            .push((key.to_string(), StateCell { value, version }));
        self
    }

    fn find_tool(&self, path: &str) -> Option<MockTool> {
        self.tools
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| path == p.as_str() || path.ends_with(p.as_str()))
            .map(|(_, t)| MockTool {
                ok: t.ok,
                value: t.value.clone(),
            })
            .or_else(|| {
                // Default: echo the first arg back as a string, ok.
                Some(MockTool {
                    ok: true,
                    value: Value::String(format!("mock:{}", path)),
                })
            })
    }
}

impl Host for MockHost {
    fn infer(&self, model: &str, req: InferRequest) -> Result<InferResult, HostError> {
        let canned = self
            .models
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| m == model)
            .map(|(_, c)| MockInfer {
                json: c.json.clone(),
                confidence: c.confidence,
                tokens: c.tokens,
                cost: c.cost,
            });
        if let Some(c) = canned {
            return Ok(InferResult {
                json: c.json,
                confidence: c.confidence,
                tokens: c.tokens,
                cost: c.cost,
            });
        }
        // Default: echo the input back as the model output, full confidence.
        let json = match req.input {
            Some(v) => to_json(v),
            None => serde_json::Value::String("mock model".into()),
        };
        Ok(InferResult {
            json,
            confidence: 1.0,
            tokens: 10,
            cost: 0.001,
        })
    }

    fn call_tool(&self, path: &str, _args: Vec<(String, Value)>) -> Result<ToolResult, HostError> {
        let t = self.find_tool(path).unwrap();
        Ok(ToolResult {
            ok: t.ok,
            value: t.value,
        })
    }

    fn state_read(&self, key: &str) -> Result<StateCell, HostError> {
        let cell = self
            .state
            .lock()
            .unwrap()
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, c)| c.clone())
            .unwrap_or(StateCell {
                value: Value::Null,
                version: 0,
            });
        Ok(cell)
    }

    fn state_update(
        &self,
        key: &str,
        expected_version: Option<i64>,
        value: Value,
    ) -> Result<StateCell, HostError> {
        let mut state = self.state.lock().unwrap();
        let entry = state.iter_mut().find(|(k, _)| k == key);
        let next = match entry {
            Some((_, cell)) => {
                if let Some(ev) = expected_version {
                    if ev != cell.version {
                        return Err(HostError(format!(
                            "state update conflict: expected version {}, found {}",
                            ev, cell.version
                        )));
                    }
                }
                cell.version += 1;
                cell.value = value.clone();
                cell.clone()
            }
            None => {
                let cell = StateCell {
                    value: value.clone(),
                    version: 1,
                };
                state.push((key.to_string(), cell.clone()));
                cell
            }
        };
        Ok(next)
    }

    fn record_trace(&self, label: &str, fields: Vec<(String, Value)>) {
        self.traces
            .lock()
            .unwrap()
            .push((label.to_string(), Value::Record(fields)));
    }

    fn replay_trace(&self, label: &str) -> Option<Value> {
        self.traces
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|(l, _)| l == label)
            .map(|(_, v)| v.clone())
    }

    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

// Silence an unused-import lint surface while keeping `OnceLock` available for
// future stateful host extensions.
fn _use_once_lock() {
    let _ = OnceLock::<()>::new();
}
