# Act

A programming language for agents that makes action, uncertainty, authority, and
evidence part of the type system.

Not a plan DSL. Not YAML. Not Python with a policy wrapper. Act is a real
programming language — functions, tasks, loops, libraries, modules, types,
errors, async, parallelism — where the things agents are bad at accidentally
screwing up become compiler-visible: effects, authority, uncertainty, evidence,
budgets, secrets, and state.

```
TypeScript for agents, Rust for side effects,
Temporal for execution, Rego for authority,
built around model calls as first-class expressions.
```

## Status

Pre-alpha. The kernel is being built incrementally. Nothing runs end to end yet.

## Kernel scope (v1)

- Modules and versioned imports
- Records, enums, unions
- `Option<T>`, `Result<T, E>`, refinement types
- Pure `fn`, effectful `proc`, durable `task`, reactive `agent`
- Extern tool declarations (typed, effectful, capability-gated, retryable)
- Typed model `infer` and `decide` (uncertainty-bearing)
- Effect rows, static effect checking
- Capabilities (lexical, unforgeable, attenuable)
- Budgets (wall time, tokens, cost, tool calls)
- Bounded `for` / `await all` / `await map`
- Typed durable state cells
- `try` / `match` / `recover` error handling
- `trace` / `check` / `require` / `ensure`
- Typed holes (`??`) with structured diagnostics
- Canonical formatter with stable AST node IDs
- JSON diagnostics designed for repair by agents
- Compilation to an executable graph IR

Explicitly cut from v1: macros, inheritance, operator overloading, reflection,
runtime eval, raw sockets, dynamic imports, unbounded effectful loops, shared
mutable memory between agents.

## Layout

```
crates/
  act-syntax       lexer + canonical AST
  act-parser       parser -> AST
  act-diagnostics  structured JSON diagnostics
  act-check        type, effect, capability, taint, budget checking
  act-fmt          canonical formatter (AST -> source, idempotent)
  act-ir           lowering to executable graph IR
  actc             CLI
```

## What the compiler enforces

| Code | Rule |
|------|------|
| `E_EFFECT_MISSING` | A tool/model call requires an effect not declared in scope |
| `E_UNBOUNDED_LOOP` | An effectful `while` loop has no `max` bound |
| `E_CHECK_UNHANDLED` | `check` without `else` bypasses the typed error enum |
| `E_SECRET_LEAK` | A `Secret<T>` value flows into a model `infer` input |
| `E_STATE_UPDATE_UNGUARDED` | `state.update` without an `expected_version:` guard can clobber concurrent writes |
| `E_COMPENSATION_MISSING` | A non-idempotent write inside a budgeted task has no `defer compensate` |
| `E_POLICY_MAY_UNGRANTED` | `policy_expect may X` but no matching capability is granted |
| `E_POLICY_MUST_NOT_GRANTED` | `policy_expect must_not X` but capability `X` IS granted |
| `E_REPLAY_WITHOUT_TRACE` | `replay trace("X")` references a trace that is never recorded |
| `E_HOLE_UNFILLED` | A typed hole `??` was not filled |
| `W_MODEL_CONFIDENCE_HIGH_THRESHOLD` | Model confidence threshold >= 0.90 (unreliable self-report) |

## Build & test

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings   # enforced in CI
cargo fmt --all -- --check                  # enforced in CI
```

## CLI

```sh
actc lex   <file.act>   # lex, print tokens
actc parse <file.act>   # parse, print a module summary
actc check <file.act>   # parse + check, print JSON diagnostics
actc lower <file.act>   # parse + check + lower to graph IR, print JSON
actc fmt   <file.act>   # parse + format to canonical source
```

## Example

`examples/fix_regression.act` is an end-to-end task: it fetches logs and a diff
in parallel, infers root-cause hypotheses, decides the best patch by weighted
score, opens a pull request, and records a trace an eval can replay.

```sh
actc check examples/fix_regression.act   # {"ok": true, "diagnostics": []}
actc fmt   examples/fix_regression.act   # canonicalized source
```

## License

MIT.
