use act_fmt::format_module;
use act_parser::parse_module;

/// parse → format → parse → format must be stable (idempotent).
fn assert_idempotent(src: &str) {
    let m1 = parse_module(src, 1).expect("first parse");
    let text1 = format_module(&m1);
    let m2 = match parse_module(&text1, 1) {
        Ok(m) => m,
        Err(e) => panic!(
            "re-parse failed: {} at {:?}\n--- formatted ---\n{}",
            e.message, e.span, text1
        ),
    };
    let text2 = format_module(&m2);
    assert_eq!(
        text1, text2,
        "not idempotent\n--- text1 ---\n{}\n--- text2 ---\n{}",
        text1, text2
    );
}

#[test]
fn idempotent_smoke() {
    assert_idempotent(include_str!("../../../examples/smoke.act"));
}

#[test]
fn idempotent_fix_regression() {
    assert_idempotent(include_str!("../../../examples/fix_regression.act"));
}

#[test]
fn idempotent_types() {
    assert_idempotent(
        r#"
module test@0.1

type Repo = {
  owner: String,
  name: String,
  note?: String,
}

type Score = Decimal where 0.0 <= self

type Result<T, E> =
  | ok(value: T)
  | err(error: E)
  | empty
"#,
    );
}

#[test]
fn idempotent_task_with_clauses() {
    assert_idempotent(
        r#"
module test@0.1
use tool github@2 as gh

task fetch(repo: Repo) -> String
  effects [gh.read, gh.write]
  needs [
    cap gh.pull_request.create(repo),
  ]
  budget {
    wall_time <= 30m,
    tokens <= 100,
  }
  policy_expect {
    may gh.create_pull_request
    must_not gh.merge_pull_request where true
    require_human when true
  }
{
  return ok("done")
}
"#,
    );
}

#[test]
fn idempotent_infer_decide() {
    assert_idempotent(
        r#"
module test@0.1
use model codegen@1 as coder

type Hyp = {
  claim: String,
  confidence: Decimal,
}

task analyze(input: String) -> Hyp
  effects [model]
{
  let h = infer Hyp using coder {
    goal: "Find a claim."
    input: input
    constraints: [
      "Be concise.",
    ]
  } accept {
    confidence >= 0.5,
  } else {
    return err("low confidence")
  }

  let best = decide Hyp from h
    score by [
      0.6: confidence desc,
      0.4: claim asc,
    ]
    accept confidence >= 0.8
    else return err("no good hypothesis")

  return ok(best)
}
"#,
    );
}

#[test]
fn idempotent_parallel() {
    assert_idempotent(
        r#"
module test@0.1
use tool github@2 as gh

task parallel_fetch(repo: String) -> String
  effects [gh.read]
{
  let results = await all {
    a: gh.get_file(repo: repo, path: "a"),
    b: gh.get_file(repo: repo, path: "b"),
  }
  let mapped = await map x in results parallel 2 limit 5 {
    gh.get_file(repo: repo, path: x)
  }
  return ok(results)
}
"#,
    );
}

#[test]
fn idempotent_control_flow() {
    assert_idempotent(
        r#"
module test@0.1

task flow(xs: [Int]) -> Int {
  for x in xs limit 10 {
    var i = x
    while i < 10 max 5 {
      i = i + 1
    }
  }
  if xs.len() > 0 {
    return ok(1)
  } else {
    return ok(0)
  }
}
"#,
    );
}

#[test]
fn idempotent_match_and_ops() {
    assert_idempotent(
        r#"
module test@0.1

task compute(a: Int, b: Int) -> Int {
  let mixed = a + b * 2 - (a + b)
  let cmp = (a < b) && (b == 2) || !true
  match mixed {
    ok(v) => {
      return ok(v)
    }
    err(e) => {
      return ok(0)
    }
  }
}
"#,
    );
}

#[test]
fn idempotent_secrets_and_holes() {
    assert_idempotent(
        r#"
module test@0.1
use model codegen@1 as coder

type Out = { text: String }

task work(token: Secret<String>) -> Out
  effects [model]
{
  let safe: Public<String> = redact(token)
  let hint = ?? "describe output"
  let out = infer Out using coder {
    goal: "produce"
    input: safe
  } accept {
    confidence >= 0.4,
  }
  return ok(out)
}
"#,
    );
}
