//! HTTP host: calls real OpenAI-compatible models and dispatches `gh.*` tool
//! calls to the GitHub REST API. Credentials come from the environment
//! (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_VERIFIER_MODEL`,
//! `GITHUB_TOKEN`). When a credential is absent, the corresponding capability
//! returns a host error, so the runtime degrades gracefully instead of panicking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_openai::{
    config::OpenAIConfig as AiConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
    },
    Client,
};

use crate::host::{Host, HostError, InferRequest, InferResult, StateCell, ToolResult};
use crate::value::{to_json, Value};

/// Per-token cost in USD for cost tracking. Micros per token.
/// Defaults to gpt-4o-mini pricing (~$0.15/1M input, ~$0.60/1M output);
/// override via `OPENAI_COST_PER_1K_TOKENS` (in micros, i.e. millionths of a
/// dollar per 1K tokens).
const DEFAULT_COST_PER_1K_TOKENS_MICROS: u64 = 150;

/// Configuration for an OpenAI-compatible chat-completions endpoint.
#[derive(Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Optional separate model for the verifier call. Defaults to `model`.
    pub verifier_model: String,
    /// Cost per 1K tokens in micros (millionths of a USD).
    pub cost_per_1k_tokens_micros: u64,
}

impl OpenAiConfig {
    /// Read from the environment. Returns `None` if `OPENAI_API_KEY` is unset.
    pub fn from_env() -> Option<OpenAiConfig> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        Some(OpenAiConfig {
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            api_key,
            verifier_model: std::env::var("OPENAI_VERIFIER_MODEL")
                .unwrap_or_else(|_| model.clone()),
            cost_per_1k_tokens_micros: std::env::var("OPENAI_COST_PER_1K_TOKENS_MICROS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_COST_PER_1K_TOKENS_MICROS),
            model,
        })
    }
}

/// A real HTTP host. Model calls go through `async-openai` (the canonical Rust
/// client); `gh.*` tool calls go to the GitHub REST API via blocking `reqwest`.
/// The interpreter is synchronous, so the async model client is driven by a
/// tokio runtime held here (`Handle::block_on` from the host's sync methods).
pub struct HttpHost {
    openai: Option<OpenAiClient>,
    github_token: Option<String>,
    github_api_base: String,
    state: std::sync::Mutex<HashMap<String, StateCell>>,
    traces: std::sync::Mutex<HashMap<String, Value>>,
    start: Instant,
}

struct OpenAiClient {
    client: Arc<Client<AiConfig>>,
    model: String,
    verifier_model: String,
    cost_per_1k_tokens_micros: u64,
    rt: tokio::runtime::Runtime,
}

impl HttpHost {
    pub fn from_env() -> HttpHost {
        let openai = OpenAiConfig::from_env().map(|cfg| {
            let config = AiConfig::new()
                .with_api_key(cfg.api_key.clone())
                .with_api_base(cfg.base_url.clone());
            OpenAiClient {
                client: Arc::new(Client::with_config(config)),
                model: cfg.model,
                verifier_model: cfg.verifier_model,
                cost_per_1k_tokens_micros: cfg.cost_per_1k_tokens_micros,
                rt: tokio::runtime::Runtime::new().expect("build tokio runtime"),
            }
        });
        HttpHost {
            openai,
            github_token: std::env::var("GITHUB_TOKEN").ok(),
            github_api_base: std::env::var("GITHUB_API_BASE")
                .unwrap_or_else(|_| "https://api.github.com".to_string()),
            state: std::sync::Mutex::new(HashMap::new()),
            traces: std::sync::Mutex::new(HashMap::new()),
            start: Instant::now(),
        }
    }

    /// Build the user prompt from the goal/input/constraints. The type shape
    /// is enforced server-side via `response_format`, so the prompt only needs
    /// the task context — not a description of the JSON shape.
    fn prompt(&self, req: &InferRequest) -> String {
        let mut s = String::new();
        if let Some(g) = req.goal {
            s.push_str("Goal: ");
            s.push_str(&json_to_text(g));
            s.push('\n');
        }
        if let Some(i) = req.input {
            s.push_str("Input: ");
            s.push_str(&serde_json::to_string_pretty(&to_json(i)).unwrap_or_default());
            s.push('\n');
        }
        if !req.constraints.is_empty() {
            s.push_str("Constraints:\n");
            for c in req.constraints {
                s.push_str("- ");
                s.push_str(&json_to_text(c));
                s.push('\n');
            }
        }
        s.push_str("Respond with valid JSON only.");
        s
    }
}

fn json_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(&to_json(other)).unwrap_or_default(),
    }
}

/// Compute cost in USD from token count and per-1K-token rate (in micros).
fn cost_for_tokens(per_1k_micros: u64, tokens: u64) -> f64 {
    (tokens as f64 / 1000.0) * (per_1k_micros as f64 / 1_000_000.0)
}

/// Send a chat completion request with retry on transient errors (429, 500+)
/// and a fallback if the provider doesn't support `response_format: json_schema`.
/// `use_schema` tracks whether the original request used structured output;
/// on a 400 suggesting `response_format` is unsupported, retry with the schema
/// removed (the prompt still asks for JSON).
fn chat_with_retry(
    rt: &tokio::runtime::Runtime,
    client: Arc<Client<AiConfig>>,
    request: async_openai::types::CreateChatCompletionRequest,
    use_schema: bool,
    schema: &Option<serde_json::Value>,
) -> Result<async_openai::types::CreateChatCompletionResponse, String> {
    let max_retries = 3u32;
    let mut delay = Duration::from_millis(500);
    let mut last_err = String::new();
    let mut current_req = request;
    let mut schema_dropped = false;
    for attempt in 0..=max_retries {
        let c = client.clone();
        let req = current_req.clone();
        let result = rt
            .handle()
            .block_on(async move { c.chat().create(req).await });
        match result {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                let estr = format!("{}", e);
                last_err = estr.clone();
                // If the provider rejected `response_format: json_schema`
                // (common for older / non-OpenAI providers), retry without it.
                if use_schema
                    && !schema_dropped
                    && (estr.contains("response_format") || estr.contains("400"))
                {
                    current_req.response_format = None;
                    schema_dropped = true;
                    continue;
                }
                // Retry on rate-limit (429) or server errors (5xx).
                let should_retry = estr.contains("429")
                    || estr.contains("500")
                    || estr.contains("502")
                    || estr.contains("503")
                    || estr.contains("504");
                if should_retry && attempt < max_retries {
                    std::thread::sleep(delay);
                    delay *= 2;
                    continue;
                }
                return Err(last_err);
            }
        }
    }
    let _ = schema;
    Err(last_err)
}

impl Host for HttpHost {
    fn infer(&self, model: &str, req: InferRequest) -> Result<InferResult, HostError> {
        let setup = self
            .openai
            .as_ref()
            .ok_or_else(|| HostError("no model credentials: OPENAI_API_KEY unset".into()))?;
        // The Act `using <alias>` handle is a source-level handle, not a model
        // ID. The concrete model comes from OPENAI_MODEL — except the verifier
        // alias, which routes to OPENAI_VERIFIER_MODEL so the self-hosted
        // accept-gate verifier (an ordinary `infer` in `builtin/verify.act`)
        // can use a separate model. (A future `extern model` registry could
        // resolve arbitrary aliases; until then sending an alias would 400.)
        let model_name = if model == "verifier" {
            setup.verifier_model.clone()
        } else {
            setup.model.clone()
        };
        let prompt = self.prompt(&req);
        let system = ChatCompletionRequestSystemMessageArgs::default()
            .content("You are a structured-output agent. Always respond with JSON only.")
            .build()
            .map_err(|e| HostError(format!("build system message: {}", e)))?
            .into();
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()
            .map_err(|e| HostError(format!("build user message: {}", e)))?
            .into();
        // If the host provided a JSON Schema for the target type, use
        // `response_format: { type: "json_schema", strict: true }` so the
        // provider guarantees the shape server-side. This eliminates the
        // silent field-drop coercion bug. Not all providers support this;
        // a 400 on this field means the provider doesn't — fall back to
        // no schema on retry (the prompt still asks for JSON).
        let schema = req.ty_schema.cloned();
        let use_schema = schema.is_some();
        let response_format = schema.as_ref().map(|s| ResponseFormat::JsonSchema {
            json_schema: ResponseFormatJsonSchema {
                name: "act_infer_output".to_string(),
                description: None,
                schema: Some(s.clone()),
                strict: Some(true),
            },
        });
        let mut request = CreateChatCompletionRequestArgs::default()
            .model(&model_name)
            .messages([system, user])
            .logprobs(true)
            .build()
            .map_err(|e| HostError(format!("build chat request: {}", e)))?;
        request.response_format = response_format;
        let client = setup.client.clone();
        let response = chat_with_retry(&setup.rt, client, request, use_schema, &schema)
            .map_err(|e| HostError(format!("model request failed: {}", e)))?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(HostError("model returned no choice".into()))?;
        let content = choice
            .message
            .content
            .ok_or(HostError("model returned no content".into()))?;
        let tokens = response.usage.map(|u| u.total_tokens as u64).unwrap_or(0);
        // Token-logprob confidence is a fluency proxy, not correctness. The
        // verifier gate (see `verify`) is the real signal; this is kept as a
        // secondary heuristic.
        let confidence = choice
            .logprobs
            .and_then(|lp| lp.content)
            .map(|toks| {
                if toks.is_empty() {
                    return 1.0;
                }
                let mean_logprob: f64 =
                    toks.iter().map(|t| t.logprob as f64).sum::<f64>() / toks.len() as f64;
                mean_logprob.exp()
            })
            .unwrap_or(1.0);
        let json = serde_json::from_str::<serde_json::Value>(&content)
            .unwrap_or(serde_json::Value::String(content));
        let cost = cost_for_tokens(setup.cost_per_1k_tokens_micros, tokens);
        Ok(InferResult {
            json,
            confidence,
            tokens,
            cost,
        })
    }

    fn call_tool(&self, path: &str, args: Vec<(String, Value)>) -> Result<ToolResult, HostError> {
        if let Some(rest) = path.strip_prefix("gh.") {
            return self.github(rest, &args);
        }
        if let Some(rest) = path.strip_prefix("eo.") {
            return self.evalops(rest, &args);
        }
        Err(HostError(format!(
            "no HTTP dispatch configured for tool `{}`",
            path
        )))
    }

    fn state_read(&self, key: &str) -> Result<StateCell, HostError> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .get(key)
            .cloned()
            .unwrap_or(StateCell {
                value: Value::Null,
                version: 0,
            }))
    }

    fn state_update(
        &self,
        key: &str,
        expected_version: Option<i64>,
        value: Value,
    ) -> Result<StateCell, HostError> {
        let mut state = self.state.lock().unwrap();
        let entry = state.get_mut(key);
        let next = match entry {
            Some(cell) => {
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
                let cell = StateCell { value, version: 1 };
                state.insert(key.to_string(), cell.clone());
                cell
            }
        };
        Ok(next)
    }

    fn record_trace(&self, label: &str, fields: Vec<(String, Value)>) {
        self.traces
            .lock()
            .unwrap()
            .insert(label.to_string(), Value::Record(fields));
    }

    fn replay_trace(&self, label: &str) -> Option<Value> {
        self.traces.lock().unwrap().get(label).cloned()
    }

    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl HttpHost {
    fn github(&self, op: &str, args: &[(String, Value)]) -> Result<ToolResult, HostError> {
        let token = self
            .github_token
            .as_ref()
            .ok_or_else(|| HostError("no GitHub credentials: GITHUB_TOKEN unset".into()))?;
        let arg = |name: &str| args.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
        let repo_full = |v: &Value| -> String {
            if let Value::Record(fs) = v {
                let owner = fs.iter().find(|(n, _)| n == "owner").and_then(|(_, v)| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
                let name = fs.iter().find(|(n, _)| n == "name").and_then(|(_, v)| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
                match (owner, name) {
                    (Some(o), Some(n)) => format!("{}/{}", o, n),
                    _ => String::new(),
                }
            } else if let Value::String(s) = v {
                s.clone()
            } else {
                String::new()
            }
        };
        let headers = vec![
            ("Authorization", format!("Bearer {}", token)),
            ("Accept", "application/vnd.github+json".to_string()),
            ("X-GitHub-Api-Version", "2022-11-28".to_string()),
        ];
        match op {
            "create_pull_request" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let base = text_arg(args, "base");
                let base = if base.is_empty() {
                    "main".to_string()
                } else {
                    base
                };
                let body = serde_json::json!({
                    "title": text_arg(args, "title"),
                    "head": text_arg(args, "branch"),
                    "base": base,
                    "body": text_arg(args, "body"),
                });
                let url = format!("{}/repos/{}/pulls", self.github_api_base, repo_full(&repo));
                let resp = blocking_post_json(&url, &body, &headers)?;
                let html = resp
                    .get("html_url")
                    .and_then(|u| u.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(html),
                })
            }
            "close_pull_request" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let pr_number = text_arg(args, "number");
                if pr_number.is_empty() {
                    return Err(HostError("close_pull_request requires `number`".into()));
                }
                let url = format!(
                    "{}/repos/{}/pulls/{}",
                    self.github_api_base,
                    repo_full(&repo),
                    pr_number
                );
                let body = serde_json::json!({"state": "closed"});
                let _ = blocking_patch_json(&url, &body, &headers)?;
                Ok(ToolResult {
                    ok: true,
                    value: Value::String("closed".to_string()),
                })
            }
            "get_file" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let path = text_arg(args, "path");
                let url = format!(
                    "{}/repos/{}/contents/{}",
                    self.github_api_base,
                    repo_full(&repo),
                    path
                );
                let resp = blocking_get_json(&url, &headers)?;
                let content = resp
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.replace('\n', ""))
                    .and_then(|s| base64_decode(&s))
                    .unwrap_or_default();
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(content),
                })
            }
            "compare" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let base = text_arg(args, "base");
                let head = text_arg(args, "head");
                let url = format!(
                    "{}/repos/{}/compare/{}...{}",
                    self.github_api_base,
                    repo_full(&repo),
                    base,
                    head
                );
                let resp = blocking_get_json(&url, &headers)?;
                // Return the patch/diff text, not just the html_url — the
                // model needs to see what changed.
                let diff = resp
                    .get("files")
                    .and_then(|files| files.as_array())
                    .map(|files| {
                        files
                            .iter()
                            .map(|f| {
                                let filename =
                                    f.get("filename").and_then(|v| v.as_str()).unwrap_or("");
                                let patch = f.get("patch").and_then(|v| v.as_str()).unwrap_or("");
                                format!("--- {}\n{}", filename, patch)
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_else(|| {
                        resp.get("html_url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("compare")
                            .to_string()
                    });
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(diff),
                })
            }
            "get_logs" => {
                // Fetch workflow run logs via the GitHub Actions API.
                // GET /repos/{owner}/{repo}/actions/runs/{run_id}/logs returns
                // a redirect to a zip URL; we return the run's job logs as
                // a concatenated string.
                let repo = arg("repo").unwrap_or(Value::Null);
                let run_id = text_arg(args, "id");
                if run_id.is_empty() {
                    return Err(HostError("get_logs requires `id` (run_id)".into()));
                }
                let url = format!(
                    "{}/repos/{}/actions/runs/{}/jobs",
                    self.github_api_base,
                    repo_full(&repo),
                    run_id
                );
                let resp = blocking_get_json(&url, &headers)?;
                let logs = resp
                    .get("jobs")
                    .and_then(|jobs| jobs.as_array())
                    .map(|jobs| {
                        jobs.iter()
                            .map(|job| {
                                let name =
                                    job.get("name").and_then(|v| v.as_str()).unwrap_or("job");
                                let conclusion = job
                                    .get("conclusion")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("?");
                                let steps = job
                                    .get("steps")
                                    .and_then(|s| s.as_array())
                                    .map(|steps| {
                                        steps
                                            .iter()
                                            .map(|s| {
                                                let sn = s
                                                    .get("name")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("");
                                                let sc = s
                                                    .get("conclusion")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("?");
                                                format!("  - [{}] {}", sc, sn)
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    })
                                    .unwrap_or_default();
                                format!("[{}] {}\n{}", conclusion, name, steps)
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .unwrap_or_default();
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(logs),
                })
            }
            _ => Err(HostError(format!("unsupported GitHub op `{}`", op))),
        }
    }

    /// Dispatch `eo.*` (evalops) tool calls. These read CI/test results from
    /// the GitHub Actions API — `fetch_logs` and `failing_tests` map to the
    /// same run-jobs endpoint, just sliced differently.
    fn evalops(&self, op: &str, args: &[(String, Value)]) -> Result<ToolResult, HostError> {
        let token = self.github_token.as_ref().ok_or_else(|| {
            HostError("no GitHub credentials for eo.*: GITHUB_TOKEN unset".into())
        })?;
        let arg = |name: &str| args.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
        let repo_full = |v: &Value| -> String {
            if let Value::Record(fs) = v {
                let owner = fs.iter().find(|(n, _)| n == "owner").and_then(|(_, v)| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
                let name = fs.iter().find(|(n, _)| n == "name").and_then(|(_, v)| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                });
                match (owner, name) {
                    (Some(o), Some(n)) => format!("{}/{}", o, n),
                    _ => String::new(),
                }
            } else if let Value::String(s) = v {
                s.clone()
            } else {
                String::new()
            }
        };
        let headers = vec![
            ("Authorization", format!("Bearer {}", token)),
            ("Accept", "application/vnd.github+json".to_string()),
            ("X-GitHub-Api-Version", "2022-11-28".to_string()),
        ];
        match op {
            "fetch_logs" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let run_id = text_arg(args, "run_id");
                if run_id.is_empty() {
                    return Err(HostError("fetch_logs requires `run_id`".into()));
                }
                let url = format!(
                    "{}/repos/{}/actions/runs/{}/jobs",
                    self.github_api_base,
                    repo_full(&repo),
                    run_id
                );
                let resp = blocking_get_json(&url, &headers)?;
                let logs = resp
                    .get("jobs")
                    .and_then(|jobs| jobs.as_array())
                    .map(|jobs| {
                        jobs.iter()
                            .map(|job| {
                                let name =
                                    job.get("name").and_then(|v| v.as_str()).unwrap_or("job");
                                let conclusion = job
                                    .get("conclusion")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("?");
                                format!("[{}] {}", conclusion, name)
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(logs),
                })
            }
            "failing_tests" => {
                let repo = arg("repo").unwrap_or(Value::Null);
                let run_id = text_arg(args, "run_id");
                if run_id.is_empty() {
                    return Err(HostError("failing_tests requires `run_id`".into()));
                }
                let url = format!(
                    "{}/repos/{}/actions/runs/{}/jobs",
                    self.github_api_base,
                    repo_full(&repo),
                    run_id
                );
                let resp = blocking_get_json(&url, &headers)?;
                let failures: Vec<String> = resp
                    .get("jobs")
                    .and_then(|jobs| jobs.as_array())
                    .map(|jobs| {
                        jobs.iter()
                            .flat_map(|job| {
                                job.get("steps")
                                    .and_then(|s| s.as_array())
                                    .map(|steps| {
                                        steps
                                            .iter()
                                            .filter(|s| {
                                                s.get("conclusion").and_then(|v| v.as_str())
                                                    == Some("failure")
                                            })
                                            .map(|s| {
                                                s.get("name")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("unknown")
                                                    .to_string()
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default()
                                    .into_iter()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(ToolResult {
                    ok: true,
                    value: Value::Array(
                        failures.iter().map(|s| Value::String(s.clone())).collect(),
                    ),
                })
            }
            _ => Err(HostError(format!("unsupported evalops op `{}`", op))),
        }
    }
}

fn text_arg(args: &[(String, Value)], name: &str) -> String {
    args.iter()
        .find(|(n, _)| n == name)
        .and_then(|(_, v)| {
            if let Value::String(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn blocking_post_json(
    url: &str,
    body: &serde_json::Value,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, HostError> {
    blocking_send("POST", url, Some(body), headers)
}

fn blocking_patch_json(
    url: &str,
    body: &serde_json::Value,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, HostError> {
    blocking_send("PATCH", url, Some(body), headers)
}

fn blocking_get_json(
    url: &str,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, HostError> {
    blocking_send("GET", url, None, headers)
}

fn blocking_send(
    method: &str,
    url: &str,
    body: Option<&serde_json::Value>,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, HostError> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("act-runtime/0.1")
        .build()
        .map_err(|e| HostError(format!("http client build failed: {}", e)))?;
    let mut req = match method {
        "POST" => client.post(url),
        "PATCH" => client.patch(url),
        "GET" => client.get(url),
        other => return Err(HostError(format!("unsupported method: {}", other))),
    };
    if let Some(b) = body {
        req = req.json(b);
    }
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    let resp = req
        .send()
        .map_err(|e| HostError(format!("request failed: {}", e)))?;
    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| HostError(format!("read body: {}", e)))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| HostError(format!("decode json: {} (body: {})", e, text)))?;
    if !status.is_success() {
        return Err(HostError(format!("HTTP {}: {}", status, json)));
    }
    Ok(json)
}

/// Minimal RFC 4648 base64 decode (standard alphabet, no padding required).
fn base64_decode(s: &str) -> Option<String> {
    let s = s.trim_end_matches('=');
    let mut buf = 0u32;
    let mut bits = 0u32;
    let mut out = Vec::new();
    for c in s.chars() {
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '-' => 62,
            '/' => 63,
            '_' => 63,
            _ => return None,
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    String::from_utf8(out).ok()
}
