//! Wire-level integration test: runs a task against a *real* local HTTP server
//! (not the in-process MockHost). This proves the model client (async-openai)
//! and the GitHub tool dispatcher actually round-trip over HTTP, not merely
//! that the wiring exists. Credentials point at 127.0.0.1; no network egress.
//!
//! Three HTTP exchanges are served: (1) the model `infer` call, (2) the
//! verifier `verify` call (second model call for the accept gate), and (3) the
//! GitHub `create_pull_request` call.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use act_parser::parse_module;
use act_run::{run_task, HttpHost, RunConfig, Value};

const TASK_SRC: &str = r#"
module wire@0.1
use model codegen@1 as coder

type Repo = {
  owner: String,
  name: String,
}

type Draft = {
  title: String,
  body: String,
}

task run(repo: Repo, branch: String) -> Result<String, String>
  effects [gh.read, gh.write, model]
{
  let draft = infer Draft using coder {
    goal: "draft"
    input: branch
  } accept {
    confidence >= 0.5,
  }
  let url = try gh.create_pull_request(repo: repo, branch: branch, title: draft.title, body: draft.body)
  return ok(url)
}
"#;

/// Canned model response: JSON for `Draft` with a single token logprob of -0.1.
const CHAT_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":0,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"{\"title\":\"fix\",\"body\":\"patch\"}"},"logprobs":{"content":[{"token":"x","logprob":-0.1,"bytes":[120],"top_logprobs":[]}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

/// Canned verifier response: confidence 0.9 (passes the >= 0.5 gate).
const VERIFY_RESPONSE: &str = r#"{"id":"chatcmpl-2","object":"chat.completion","created":0,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"{\"confidence\":0.9,\"reason\":\"looks good\"}"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

const PR_RESPONSE: &str = r#"{"html_url":"https://github.com/evalops/act/pull/42"}"#;

fn handle(mut stream: TcpStream, chat_count: Arc<AtomicUsize>) {
    let mut reader = BufReader::new(&mut stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            break;
        }
        if line == "\r\n" || line.is_empty() {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            content_length = line
                .split(':')
                .nth(1)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
        }
    }
    if content_length > 0 {
        let mut body = vec![0u8; content_length];
        let _ = reader.read_exact(&mut body);
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let body = if path.contains("/chat/completions") {
        // First chat call is the infer, second is the verify.
        let n = chat_count.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            CHAT_RESPONSE
        } else {
            VERIFY_RESPONSE
        }
    } else if path.contains("/pulls") {
        PR_RESPONSE
    } else {
        "{}"
    };
    let out = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(out.as_bytes());
}

/// Serve HTTP exchanges on a kernel-assigned port, then stop.
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        listener.set_nonblocking(false).ok();
        let chat_count = Arc::new(AtomicUsize::new(0));
        // Three requests: model infer, verifier, GitHub PR.
        for _ in 0..3 {
            match listener.accept() {
                Ok((stream, _)) => handle(stream, chat_count.clone()),
                Err(_) => break,
            }
        }
    });
    format!("http://{}", addr)
}

#[test]
fn round_trips_over_http() {
    let base = serve();
    // Point the HTTP host at the local server.
    std::env::set_var("OPENAI_API_KEY", "test-key");
    std::env::set_var("OPENAI_BASE_URL", &base);
    std::env::set_var("OPENAI_VERIFIER_MODEL", "test-verifier");
    std::env::set_var("GITHUB_TOKEN", "test-token");
    std::env::set_var("GITHUB_API_BASE", &base);

    let module = parse_module(TASK_SRC, 1).expect("parse");
    let host = HttpHost::from_env();
    let cfg = RunConfig {
        host: &host,
        granted_caps: HashSet::new(),
    };
    let repo = Value::Record(vec![
        ("owner".into(), Value::String("evalops".into())),
        ("name".into(), Value::String("act".into())),
    ]);
    let result = run_task(
        &module,
        "run",
        vec![
            ("repo".into(), repo),
            ("branch".into(), Value::String("agent/draft".into())),
        ],
        &cfg,
    )
    .expect("run should succeed");

    // The model output flowed through async-openai; the verifier confirmed it;
    // the GitHub call through reqwest; the task unwrapped all and returned the
    // PR url.
    let url = match result {
        Value::Result {
            ok: true,
            value: Some(v),
        } => *v,
        other => panic!("expected ok result, got {:?}", other),
    };
    match url {
        Value::String(s) => assert_eq!(s, "https://github.com/evalops/act/pull/42"),
        other => panic!("expected url string, got {:?}", other),
    }
}
