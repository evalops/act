//! HTTP host: calls real OpenAI-compatible models and dispatches `gh.*` tool
//! calls to the GitHub REST API. Credentials come from the environment
//! (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `GITHUB_TOKEN`).
//! When a credential is absent, the corresponding capability returns a host
//! error, so the runtime degrades gracefully instead of panicking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_openai::{
    config::OpenAIConfig as AiConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};

use crate::host::{Host, HostError, InferRequest, InferResult, StateCell, ToolResult};
use crate::value::{to_json, Value};

/// Configuration for an OpenAI-compatible chat-completions endpoint.
#[derive(Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiConfig {
    /// Read from the environment. Returns `None` if `OPENAI_API_KEY` is unset.
    pub fn from_env() -> Option<OpenAiConfig> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        Some(OpenAiConfig {
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            api_key,
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
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
        s.push_str("Respond with only valid JSON for the requested type.");
        s
    }
}

fn json_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(&to_json(other)).unwrap_or_default(),
    }
}

impl Host for HttpHost {
    fn infer(&self, model: &str, req: InferRequest) -> Result<InferResult, HostError> {
        let setup = self
            .openai
            .as_ref()
            .ok_or_else(|| HostError("no model credentials: OPENAI_API_KEY unset".into()))?;
        // The Act `using <alias>` handle is not a real model name; the concrete
        // model comes from OPENAI_MODEL (unless the source names one explicitly).
        let model_name = if model.is_empty() {
            setup.model.clone()
        } else {
            model.to_string()
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
        let request = CreateChatCompletionRequestArgs::default()
            .model(&model_name)
            .messages([system, user])
            // Request token logprobs so the accept gate has a real signal
            // (mean chosen-token probability) instead of an always-pass placeholder.
            .logprobs(true)
            .build()
            .map_err(|e| HostError(format!("build chat request: {}", e)))?;
        let client = setup.client.clone();
        let response = setup
            .rt
            .handle()
            .block_on(async move { client.chat().create(request).await })
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
        // Confidence = geometric mean of chosen-token probabilities. This is a
        // token-level proxy, not a calibrated answer-confidence; treat it as a
        // coarse gate and prefer a verifier-derived score where it matters.
        let confidence = choice
            .logprobs
            .and_then(|lp| lp.content)
            .map(|tokens| {
                if tokens.is_empty() {
                    return 1.0;
                }
                let mean_logprob: f64 =
                    tokens.iter().map(|t| t.logprob as f64).sum::<f64>() / tokens.len() as f64;
                mean_logprob.exp()
            })
            .unwrap_or(1.0);
        // Parse the content as JSON; fall back to wrapping it as a string.
        let json = serde_json::from_str::<serde_json::Value>(&content)
            .unwrap_or(serde_json::Value::String(content));
        Ok(InferResult {
            json,
            confidence,
            tokens,
            cost: 0.0,
        })
    }

    fn call_tool(&self, path: &str, args: Vec<(String, Value)>) -> Result<ToolResult, HostError> {
        if let Some(rest) = path.strip_prefix("gh.") {
            return self.github(rest, &args);
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
                let body = serde_json::json!({
                    "title": text_arg(args, "title"),
                    "head": text_arg(args, "branch"),
                    "base": "main",
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
                Ok(ToolResult {
                    ok: true,
                    value: Value::String(
                        resp.get("html_url")
                            .and_then(|u| u.as_str())
                            .unwrap_or("compare")
                            .to_string(),
                    ),
                })
            }
            _ => Err(HostError(format!("unsupported GitHub op `{}`", op))),
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
    let client = reqwest::blocking::Client::builder()
        .user_agent("act-runtime/0.1")
        .build()
        .map_err(|e| HostError(format!("http client build failed: {}", e)))?;
    let mut req = client.post(url).json(body);
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    let resp = req
        .send()
        .map_err(|e| HostError(format!("request failed: {}", e)))?;
    let status = resp.status();
    let json: serde_json::Value = serde_json::from_str(
        &resp
            .text()
            .map_err(|e| HostError(format!("read body: {}", e)))?,
    )
    .map_err(|e| HostError(format!("decode json: {}", e)))?;
    if !status.is_success() {
        return Err(HostError(format!("HTTP {}: {}", status, json)));
    }
    Ok(json)
}

fn blocking_get_json(
    url: &str,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, HostError> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("act-runtime/0.1")
        .build()
        .map_err(|e| HostError(format!("http client build failed: {}", e)))?;
    let mut req = client.get(url);
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    let resp = req
        .send()
        .map_err(|e| HostError(format!("request failed: {}", e)))?;
    let status = resp.status();
    let json: serde_json::Value = serde_json::from_str(
        &resp
            .text()
            .map_err(|e| HostError(format!("read body: {}", e)))?,
    )
    .map_err(|e| HostError(format!("decode json: {}", e)))?;
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
