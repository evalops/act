use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: actc <command> [args]");
        eprintln!("commands:");
        eprintln!("  check <file.act>       parse + check, print diagnostics as JSON");
        eprintln!("  parse <file.act>       parse only, print AST summary");
        eprintln!("  lower <file.act>       parse + check + lower to graph IR, print JSON");
        eprintln!("  fmt   <file.act>       parse + format to canonical source");
        eprintln!("  run   <file.act> <task> [name=json ...]  parse + check + execute");
        eprintln!("                                     also: --args-json '{{\"k\":v}}' or --args-file f.json");
        eprintln!("                                     needs OPENAI_API_KEY, GITHUB_TOKEN for real calls");
        eprintln!("  lex   <file.act>       lex only, print tokens");
        eprintln!("  repl                    interactive REPL (mock host, :help for commands)");
        return ExitCode::from(2);
    }
    let cmd = args[1].as_str();
    // `repl` doesn't need a file path.
    if cmd == "repl" {
        return repl();
    }
    let path = match args.get(2) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("error: missing file path");
            return ExitCode::from(2);
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::from(2);
        }
    };
    let file_id = 1u32;
    match cmd {
        "lex" => match act_syntax::lex(&src, file_id) {
            Ok(toks) => {
                for t in toks {
                    println!(
                        "{:?}\t{:?}\t{}",
                        t.span,
                        t.kind,
                        t.text.replace('\n', "\\n")
                    );
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("lex error: {} at {:?}", e.message, e.span);
                ExitCode::from(1)
            }
        },
        "parse" => match act_parser::parse_module(&src, file_id) {
            Ok(m) => {
                print_module(&m);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("parse error: {} at {:?}", e.message, e.span);
                ExitCode::from(1)
            }
        },
        "check" => {
            let module = match act_parser::parse_module(&src, file_id) {
                Ok(m) => m,
                Err(e) => {
                    let diag = act_diagnostics::Diagnostic::new(
                        act_diagnostics::codes::E_PARSE,
                        act_diagnostics::Severity::Error,
                        act_syntax::ast::Span {
                            file: e.span.file,
                            start: e.span.start,
                            end: e.span.end,
                        },
                        e.message,
                    );
                    let report = act_diagnostics::DiagnosticReport::new(vec![diag]);
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                    return ExitCode::from(1);
                }
            };
            let out = act_check::check(&module);
            println!("{}", serde_json::to_string_pretty(&out.report).unwrap());
            if out.report.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "lower" => {
            let module = match act_parser::parse_module(&src, file_id) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("parse error: {} at {:?}", e.message, e.span);
                    return ExitCode::from(1);
                }
            };
            let chk = act_check::check(&module);
            if !chk.report.ok {
                println!("{}", serde_json::to_string_pretty(&chk.report).unwrap());
                return ExitCode::from(1);
            }
            let out = act_ir::lower(&module);
            if let Some(g) = out.graph {
                println!("{}", serde_json::to_string_pretty(&g).unwrap());
                ExitCode::SUCCESS
            } else {
                println!("{}", serde_json::to_string_pretty(&out.report).unwrap());
                ExitCode::from(1)
            }
        }
        "fmt" => match act_parser::parse_module(&src, file_id) {
            Ok(m) => {
                print!("{}", act_fmt::format_module(&m));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("parse error: {} at {:?}", e.message, e.span);
                ExitCode::from(1)
            }
        },
        "run" => {
            let task = match args.get(3) {
                Some(t) => t.clone(),
                None => {
                    eprintln!("usage: actc run <file.act> <task> [name=json ...]");
                    eprintln!("       actc run <file.act> <task> --args-json '{{\"k\":v}}'");
                    eprintln!("       actc run <file.act> <task> --args-file args.json");
                    return ExitCode::from(2);
                }
            };
            let module = match act_parser::parse_module(&src, file_id) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("parse error: {} at {:?}", e.message, e.span);
                    return ExitCode::from(1);
                }
            };
            let chk = act_check::check(&module);
            if !chk.report.ok {
                println!("{}", serde_json::to_string_pretty(&chk.report).unwrap());
                return ExitCode::from(1);
            }
            // Parse args from: --args-json '{...}', --args-file path, or
            // positional name=json pairs.
            let mut run_args: Vec<(String, act_run::Value)> = Vec::new();
            let mut i = 4;
            while i < args.len() {
                let a = &args[i];
                if a == "--args-json" {
                    if let Some(json_str) = args.get(i + 1) {
                        let json: serde_json::Value = serde_json::from_str(json_str)
                            .unwrap_or_else(|e| {
                                eprintln!("error: --args-json is not valid JSON: {}", e);
                                std::process::exit(2);
                            });
                        if let serde_json::Value::Object(map) = json {
                            for (k, v) in map {
                                run_args.push((k.clone(), act_run::value_from_json(&v)));
                            }
                        }
                        i += 2;
                        continue;
                    }
                }
                if a == "--args-file" {
                    if let Some(file_path) = args.get(i + 1) {
                        let file_content = std::fs::read_to_string(file_path).unwrap_or_else(|e| {
                            eprintln!("error: cannot read args file: {}", e);
                            std::process::exit(2);
                        });
                        let json: serde_json::Value = serde_json::from_str(&file_content)
                            .unwrap_or_else(|e| {
                                eprintln!("error: args file is not valid JSON: {}", e);
                                std::process::exit(2);
                            });
                        if let serde_json::Value::Object(map) = json {
                            for (k, v) in map {
                                run_args.push((k.clone(), act_run::value_from_json(&v)));
                            }
                        }
                        i += 2;
                        continue;
                    }
                }
                // Positional name=json pair.
                match a.split_once('=') {
                    Some((k, v)) => {
                        let json: serde_json::Value = serde_json::from_str(v)
                            .unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
                        let val = act_run::value_from_json(&json);
                        run_args.push((k.to_string(), val));
                    }
                    None => {
                        run_args.push((a.clone(), act_run::Value::String(a.clone())));
                    }
                }
                i += 1;
            }
            let host = act_run::HttpHost::from_env();
            let cfg = act_run::RunConfig {
                host: &host,
                granted_caps: std::collections::HashSet::new(),
            };
            match act_run::run_task(&module, &task, run_args, &cfg) {
                Ok(v) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&act_run::to_json(&v)).unwrap()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("run error: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        other => {
            eprintln!("unknown command: {}", other);
            ExitCode::from(2)
        }
    }
}

/// A minimal REPL: evaluates expressions and statements against a mock host.
/// Supports `:help`, `:type <expr>`, `:check <src>`, `:load <file.act>`, and
/// `:quit`. Everything else is wrapped in a module + task and evaluated.
fn repl() -> ExitCode {
    use std::io::{BufRead, Write};
    let host = act_run::MockHost::new();
    let cfg = act_run::RunConfig {
        host: &host,
        granted_caps: std::collections::HashSet::new(),
    };
    let stdin = std::io::stdin();
    let mut lines: Vec<String> = Vec::new();
    println!("act repl — type :help for commands, :quit to exit");
    loop {
        let prompt = if lines.is_empty() { "act> " } else { "...> " };
        print!("{}", prompt);
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            _ => {}
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !lines.is_empty() {
            lines.push(line.clone());
            if trimmed.ends_with('}') || trimmed.ends_with(')') {
                let src = lines.join("\n");
                eval_repl_input(&src, &cfg);
                lines.clear();
            }
            continue;
        }
        match trimmed {
            ":quit" | ":q" => break,
            ":help" | ":h" => {
                println!("  :help          show this help");
                println!("  :quit          exit");
                println!("  :check <src>   parse + check an Act source snippet");
                println!("  :load <file>   load and check an Act file");
                println!("  <expr>         evaluate an expression (wrapped in a task)");
                println!("  multi-line: start with `{{` or `(` to continue on next lines");
            }
            _ if trimmed.starts_with(":check ") => {
                let src = &trimmed[7..];
                check_snippet(src);
            }
            _ if trimmed.starts_with(":load ") => {
                let path = trimmed[6..].trim();
                match std::fs::read_to_string(path) {
                    Ok(src) => check_snippet(&src),
                    Err(e) => eprintln!("error: {}", e),
                }
            }
            _ => {
                eval_repl_input(trimmed, &cfg);
            }
        }
    }
    ExitCode::SUCCESS
}

fn check_snippet(src: &str) {
    let full = format!("module repl@0.1\n{}", src);
    match act_parser::parse_module(&full, 1) {
        Ok(m) => {
            let out = act_check::check(&m);
            if out.report.ok {
                println!("ok");
            } else {
                for d in &out.report.diagnostics {
                    println!("  {:?} {}", d.severity, d.message);
                }
            }
        }
        Err(e) => eprintln!("parse error: {}", e.message),
    }
}

fn eval_repl_input(input: &str, cfg: &act_run::RunConfig) {
    // Wrap the input in a task and evaluate it against the mock host.
    let src = format!(
        "module repl@0.1\ntask repl() -> Result<String, String>\n  effects []\n{{\n  return ok(json_stringify({}))\n}}",
        input
    );
    match act_parser::parse_module(&src, 1) {
        Ok(m) => {
            let chk = act_check::check(&m);
            if !chk.report.ok {
                for d in &chk.report.diagnostics {
                    eprintln!("  {:?} {}", d.severity, d.message);
                }
                return;
            }
            match act_run::run_task(&m, "repl", vec![], cfg) {
                Ok(v) => match v {
                    act_run::Value::Result {
                        ok: true,
                        value: Some(boxed),
                    } => match *boxed {
                        act_run::Value::String(s) => println!("{}", s),
                        other => println!("{:?}", other),
                    },
                    act_run::Value::Result {
                        ok: false,
                        value: Some(boxed),
                    } => match *boxed {
                        act_run::Value::String(s) => eprintln!("err: {}", s),
                        other => eprintln!("err: {:?}", other),
                    },
                    other => println!("{:?}", other),
                },
                Err(e) => eprintln!("runtime error: {}", e),
            }
        }
        Err(e) => eprintln!("parse error: {}", e.message),
    }
}

fn print_module(m: &act_syntax::ast::Module) {
    println!("module {}", path_string(&m.header.name));
    for u in &m.header.uses {
        let kind = match u.kind {
            act_syntax::ast::UseKind::Use => "use",
            act_syntax::ast::UseKind::Tool => "use tool",
            act_syntax::ast::UseKind::Lib => "use lib",
            act_syntax::ast::UseKind::Model => "use model",
        };
        let ver = u
            .version
            .as_ref()
            .map(|v| format!("@{}", v))
            .unwrap_or_default();
        let alias = u
            .alias
            .as_ref()
            .map(|a| format!(" as {}", a.node))
            .unwrap_or_default();
        println!("  {} {}{}{}", kind, path_string(&u.path), ver, alias);
    }
    for item in &m.items {
        match item {
            act_syntax::ast::Item::TypeDecl(t) => println!("  type {}", t.name.node),
            act_syntax::ast::Item::Fn(d) => {
                println!("  fn {} ({} params)", d.name.node, d.params.len())
            }
            act_syntax::ast::Item::Proc(d) => {
                println!("  proc {} ({} params)", d.name.node, d.params.len())
            }
            act_syntax::ast::Item::Task(d) => {
                println!("  task {} ({} params)", d.name.node, d.params.len())
            }
            act_syntax::ast::Item::Agent(a) => {
                println!("  agent {} ({} handlers)", a.name.node, a.handlers.len())
            }
            act_syntax::ast::Item::ExternTool(t) => {
                println!("  extern tool {}", path_string(&t.path))
            }
            act_syntax::ast::Item::ExternModel(t) => {
                println!("  extern model {}", path_string(&t.path))
            }
            act_syntax::ast::Item::Test(t) => println!("  test {}", t.label.node),
            act_syntax::ast::Item::Eval(t) => println!("  eval {}", t.label.node),
        }
    }
}

fn path_string(p: &[act_syntax::ast::Ident]) -> String {
    p.iter()
        .map(|i| i.node.as_str())
        .collect::<Vec<_>>()
        .join(".")
}
