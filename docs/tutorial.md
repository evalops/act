# Act: A Practical Tutorial

This tutorial takes you from zero to writing self-hosting agent programs in
Act. You'll learn the language by building real tasks — not toy snippets.

## Prerequisites

```sh
cargo build                # build the compiler + runtime
cargo test                 # verify everything works
```

All examples are in `examples/`. Check any of them with:

```sh
cargo run --bin actc check examples/summarize.act
```

## 1. Your first program

Act programs are modules. Every file starts with a module declaration and
optional imports. Here's `examples/summarize.act`:

```act
module summarize@0.1

use model codegen@1 as coder

type Repo = {
  owner: String,
  name: String,
}

type Summary = {
  text: String,
}

task summarize(repo: Repo, path: String) -> Result<Summary, String>
  effects [gh.read, model]
{
  let content = try gh.get_file(repo: repo, path: path)
  let summary = infer Summary using coder {
    goal: "Summarize this file in one sentence."
    input: content
  } accept {
    confidence >= 0.2,
  }
  return ok(summary)
}
```

Key things to notice:

- **Types first.** `Repo` and `Summary` are record types. The compiler uses
  them to generate a JSON Schema sent to the model, so the model's output is
  guaranteed to match — no silent field-drop bugs.
- **Effects are declared.** `effects [gh.read, model]` says this task reads
  from GitHub and calls a model. The checker enforces this: if you call
  `gh.create_pull_request` without declaring `gh.write`, it's a compile error.
- **Model calls are first-class.** `infer Summary using coder` is an
  expression. It returns a `Summary` value, not a string you parse. The
  `accept` clause is a gate — a second model call (the self-hosted verifier)
  checks the output and the gate only passes if confidence is high enough.
- **Errors are typed.** The task returns `Result<Summary, String>`. The `try`
  keyword propagates errors; `return ok(...)` wraps the success value.

Check it:

```sh
cargo run --bin actc check examples/summarize.act
# {"ok": true, "diagnostics": []}
```

## 2. Running against real services

```sh
OPENAI_API_KEY=... GITHUB_TOKEN=... \
  cargo run --bin actc run examples/summarize.act summarize \
    --args-json '{"repo":{"owner":"evalops","name":"act"}, "path":"README.md"}'
```

The runtime calls the model (via OpenAI-compatible API), fetches the file from
GitHub, runs the verifier, and returns the result as JSON.

You can also pass args as positional `name=json` pairs or from a file:

```sh
cargo run --bin actc run examples/summarize.act summarize \
  repo='{"owner":"evalops","name":"act"}' path='"README.md"'

cargo run --bin actc run examples/summarize.act summarize --args-file args.json
```

## 3. The REPL

Experiment without writing a full program:

```sh
cargo run --bin actc repl
act> 1 + 2
3
act> "hello".to_upper()
"HELLO"
act> :check task t() -> String { return ok("hi") }
ok
act> :quit
```

Commands: `:help`, `:check <src>`, `:load <file>`, `:quit`.

## 4. Parallel work with `await all`

`examples/fix_regression.act` fetches logs, a diff, and failing tests in
parallel:

```act
let results = await all {
  logs: eo.fetch_logs(run_id: input.run_id, max_bytes: 2_000_000),
  diff: gh.compare(repo: input.repo, base: input.base_sha, head: input.head_sha),
  failures: eo.failing_tests(run_id: input.run_id),
}
```

Each branch runs on a separate OS thread sharing one atomic budget counter. A
`return` inside a branch propagates out as the task's result.

## 5. Uncertainty and the verifier

Every `infer ... accept { ... }` is gated by a **self-hosted verifier** — an
Act task in `builtin/verify.act` that makes a second model call to check the
first one's output. This is the difference between "the model was fluent" and
"the model was right."

```act
let hyp = infer Hypothesis using coder {
  goal: "Find likely root causes for this regression."
  input: { logs: results.logs, diff: results.diff, failures: results.failures }
  constraints: [
    "Each hypothesis must cite concrete evidence.",
    "Return at most 5 hypotheses.",
  ]
} accept {
  confidence >= 0.65,
} else {
  return err(low_confidence("No grounded root cause hypothesis."))
}
```

If the verifier's confidence is below 0.65, the `else` block runs and the task
returns a typed error. The verifier itself records a `trace "verifier"`
checkpoint you can replay.

## 6. Deciding between options

`decide` picks the best candidate by weighted score:

```act
let best = decide PatchAttempt from patches
  score by [
    0.5: pass_rate desc,
    0.3: confidence desc,
    0.2: files_changed asc,
  ]
  accept confidence >= 0.80
  else return err(low_confidence("No patch cleared confidence threshold."))
```

## 7. Authority: capabilities and policy

Tasks declare what they're allowed to do:

```act
task fix_regression(input: Input) -> Result<FixResult, FixError>
  effects [gh.read, gh.write, eo.read, eo.write, model]
  needs [
    cap gh.pull_request.create(input.repo),
    cap gh.pull_request.read(input.repo),
  ]
  policy_expect {
    may gh.create_pull_request
    must_not gh.merge_pull_request
    must_not gh.push_branch where true
    require_human when true
  }
```

- `needs` declares capabilities. The runner grants them at runtime.
- `policy_expect` asserts what the task may and must not do. The checker
  verifies these against `needs` at compile time — `may X` requires the cap,
  `must_not X` forbids it.

## 8. Budgets

```act
budget {
  wall_time <= 30m,
  tokens <= 150_000,
  cost <= 8.00,
  tool_calls <= 100,
}
```

The runtime tracks all four with atomics. Exceeding any limit aborts the run
with a typed error. Parallel branches share one counter.

## 9. Evidence: trace and replay

```act
trace "selected_root_cause" {
  claim: best.hypothesis.claim,
  confidence: best.confidence,
  evidence: best.hypothesis.evidence,
}
```

Traces are recorded to the host. An `eval` block can replay them later:

```act
eval "replay_root" {
  let t = replay trace("root")
  require t.claim != ""
}
```

This is how evals and postmortems reconstruct decisions.

## 10. Self-hosting: Act runs Act

The most powerful pattern. `examples/self_evolve.act` writes an Act program,
checks it with its own checker, runs it with its own runtime, and verifies
each step with its own verifier — all in one task:

```act
let candidates = await all {
  a: infer Candidate using coder { ... } accept { confidence >= 0.6 },
  b: infer Candidate using coder { ... } accept { confidence >= 0.6 },
}

let evals = await map c in [candidates.a, candidates.b] parallel 2 limit 2 {
  evaluate(c)
}

let best = decide EvalVerdict from evals
  score by [0.7: score desc, 0.3: candidate.self_score desc]
  accept score >= 0.5
```

The `evaluate` proc calls `actc.diagnose` (the Rust checker as oracle) and
`act.run_task` (Act executing Act). Every `infer` is gated by the self-hosted
verifier. The task opens a PR with `defer compensate` cleanup.

## 11. Editor support

The LSP server provides diagnostics in any editor that supports LSP:

```sh
cargo run --bin actc lsp
```

For VS Code, add a server entry in your settings:

```json
{
  "languages": {
    "act": {
      "languageId": "act",
      "extensions": [".act"],
      "command": ["cargo", "run", "--bin", "actc", "lsp"]
    }
  }
}
```

## 12. Standard library

Common types are available in `builtin/std.act`:

```act
type Repo = { owner: String, name: String }
type Hash = String
type Score = Decimal where 0.0 <= self <= 1.0
type Evidence = { source: String, quote?: String, observed_at: String }
```

String and array methods: `.len()`, `.contains()`, `.starts_with()`,
`.ends_with()`, `.to_lower()`, `.to_upper()`, `.trim()`, `.join(sep)`.

JSON builtins: `json_parse(s)`, `json_stringify(v)`.

## Next steps

- Read `examples/fix_regression.act` — the full dev-loop task.
- Read `examples/self_evolve.act` — the headline self-hosting example.
- Read `examples/eval.act` — the self-hosted eval harness.
- Read `examples/check.act` — the metacircular checker.
- Run `cargo run --bin actc check` on your own `.act` file.
