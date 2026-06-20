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
        eprintln!("  lex   <file.act>       lex only, print tokens");
        return ExitCode::from(2);
    }
    let cmd = args[1].as_str();
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
        other => {
            eprintln!("unknown command: {}", other);
            ExitCode::from(2)
        }
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
