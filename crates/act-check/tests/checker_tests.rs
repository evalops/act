use act_check::check;
use act_diagnostics::{codes, Severity};
use act_parser::parse_module;

fn check_src(src: &str) -> Vec<(String, Severity)> {
    let module = parse_module(src, 1).expect("parse should succeed");
    let out = check(&module);
    out.report
        .diagnostics
        .iter()
        .map(|d| (d.code.clone(), d.severity))
        .collect()
}

fn has_diag(diags: &[(String, Severity)], code: &str) -> bool {
    diags.iter().any(|(c, _)| c == code)
}

fn has_error(diags: &[(String, Severity)], code: &str) -> bool {
    diags
        .iter()
        .any(|(c, s)| c == code && *s == Severity::Error)
}

fn has_warning(diags: &[(String, Severity)], code: &str) -> bool {
    diags
        .iter()
        .any(|(c, s)| c == code && *s == Severity::Warning)
}

#[test]
fn test_effect_missing() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task fetch(repo: String) -> String
  effects [gh.read]
{
  let pr = gh.create_pull_request(repo: repo)
  return pr
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_EFFECT_MISSING));
}

#[test]
fn test_effect_present() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task fetch(repo: String) -> String
  effects [gh.read, gh.write]
{
  let pr = gh.create_pull_request(repo: repo)
  return pr
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_EFFECT_MISSING));
}

#[test]
fn test_unbounded_loop() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task loop_test() -> String
  effects [gh.write]
{
  while true {
    gh.create_issue(title: "test")
  }
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_UNBOUNDED_LOOP));
}

#[test]
fn test_bounded_loop_ok() {
    let src = r#"
module test@0.1

task loop_test() -> String
  effects []
{
  var i = 0
  while i < 10 max 5 {
    i = i + 1
  }
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_UNBOUNDED_LOOP));
}

#[test]
fn test_check_without_else() {
    let src = r#"
module test@0.1

task verify(x: Int) -> String
  effects []
{
  check x > 0
  return ok("ok")
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_CHECK_UNHANDLED));
}

#[test]
fn test_check_with_else_ok() {
    let src = r#"
module test@0.1

type Err = | bad_value(message: String)

task verify(x: Int) -> Result<String, Err>
  effects []
{
  check x > 0 else { return err(bad_value("x must be positive")) }
  return ok("ok")
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_CHECK_UNHANDLED));
}

#[test]
fn test_model_confidence_high_threshold() {
    let src = r#"
module test@0.1
use model codegen@1 as coder

type Hypothesis = {
  claim: String,
  confidence: Decimal,
}

task analyze(data: String) -> String
  effects [model]
{
  let h = infer Hypothesis using coder {
    goal: "Find root cause."
    input: data
  } accept {
    confidence >= 0.95
  }
  return ok(h.claim)
}
"#;
    let diags = check_src(src);
    assert!(has_warning(
        &diags,
        codes::W_MODEL_CONFIDENCE_HIGH_THRESHOLD
    ));
}

#[test]
fn test_model_confidence_reasonable_threshold() {
    let src = r#"
module test@0.1
use model codegen@1 as coder

type Hypothesis = {
  claim: String,
  confidence: Decimal,
}

task analyze(data: String) -> String
  effects [model]
{
  let h = infer Hypothesis using coder {
    goal: "Find root cause."
    input: data
  } accept {
    confidence >= 0.70
  }
  return ok(h.claim)
}
"#;
    let diags = check_src(src);
    assert!(!has_warning(
        &diags,
        codes::W_MODEL_CONFIDENCE_HIGH_THRESHOLD
    ));
}

#[test]
fn test_policy_may_without_cap() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task create_pr(repo: String) -> String
  effects [gh.write]
  needs [cap gh.pull_request.create(repo)]
  policy_expect {
    may gh.merge_pull_request
  }
{
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_POLICY_MAY_UNGRANTED));
}

#[test]
fn test_policy_must_not_with_cap() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task manage(repo: String) -> String
  effects [gh.write]
  needs [
    cap gh.pull_request.create(repo),
    cap gh.pull_request.merge(repo),
  ]
  policy_expect {
    must_not gh.merge_pull_request
  }
{
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_POLICY_MUST_NOT_GRANTED));
}

#[test]
fn test_policy_consistent() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task manage(repo: String) -> String
  effects [gh.write]
  needs [
    cap gh.pull_request.create(repo),
  ]
  policy_expect {
    may gh.create_pull_request
    must_not gh.merge_pull_request
  }
{
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_POLICY_MAY_UNGRANTED));
    assert!(!has_error(&diags, codes::E_POLICY_MUST_NOT_GRANTED));
}

#[test]
fn test_compensation_missing() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task deploy(repo: String) -> String
  effects [gh.write]
  needs [cap gh.pull_request.create(repo)]
  budget { wall_time <= 30m }
{
  let pr = gh.create_pull_request(repo: repo)
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(has_error(&diags, codes::E_COMPENSATION_MISSING));
}

#[test]
fn test_compensation_with_defer() {
    let src = r#"
module test@0.1
use tool github@2 as gh

task deploy(repo: String) -> String
  effects [gh.write]
  needs [cap gh.pull_request.create(repo)]
  budget { wall_time <= 30m }
{
  let pr = gh.create_pull_request(repo: repo)
  defer compensate {
    gh.close_pull_request(repo: repo)
  }
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_COMPENSATION_MISSING));
}

#[test]
fn test_compensation_idempotent_tool() {
    let src = r#"
module test@0.1
use tool github@2 as gh

extern tool gh.create_pull_request(
  repo: String,
) -> String
  effects [gh.write]
  needs [cap gh.pull_request.create(repo)]
  idempotent by hash(repo)

task deploy(repo: String) -> String
  effects [gh.write]
  needs [cap gh.pull_request.create(repo)]
  budget { wall_time <= 30m }
{
  let pr = gh.create_pull_request(repo: repo)
  return ok("done")
}
"#;
    let diags = check_src(src);
    assert!(!has_error(&diags, codes::E_COMPENSATION_MISSING));
}

#[test]
fn test_hole_unfilled() {
    let src = r#"
module test@0.1

task with_hole(x: Int) -> String
  effects []
{
  let q: String = ?? "Find the right query."
  return ok(q)
}
"#;
    let module = parse_module(src, 1).expect("parse should succeed");
    let out = act_ir::lower(&module);
    assert!(out
        .report
        .diagnostics
        .iter()
        .any(|d| d.code == codes::E_HOLE_UNFILLED));
}

#[test]
fn test_clean_module() {
    let src = r#"
module clean@0.1
use tool github@2 as gh

type Repo = {
  owner: String,
  name: String,
}

type Err = | not_found(message: String)

fn add(a: Int, b: Int) -> Int {
  a + b
}

proc fetch(repo: Repo) -> Result<String, Err>
  effects [gh.read]
{
  let result = try gh.get_file(repo: repo, path: "README.md")
  return ok(result)
}
"#;
    let diags = check_src(src);
    assert!(
        diags.is_empty(),
        "expected no diagnostics, got: {:?}",
        diags
    );
}
