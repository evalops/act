# Act — Agent Configuration

## Build and test

```sh
cargo build          # build all crates
cargo test           # run all tests (16 checker tests)
cargo run --bin actc lex <file.act>    # lex a file, print tokens
cargo run --bin actc parse <file.act>  # parse, print AST summary
cargo run --bin actc check <file.act>  # parse + check, print JSON diagnostics
cargo run --bin actc lower <file.act>  # parse + check + lower to graph IR
```

## Architecture

- `act-syntax` — lexer + canonical AST (stable NodeIds, spans)
- `act-parser` — recursive descent parser
- `act-diagnostics` — structured JSON diagnostics with repair patches
- `act-check` — effect/capability/taint/budget/compensation/policy checking
- `act-ir` — lowering to executable graph IR
- `actc` — CLI driver

## Checker rules (what the compiler enforces)

| Code | Rule |
|------|------|
| E_EFFECT_MISSING | Call requires an effect not declared in scope |
| E_UNBOUNDED_LOOP | Effectful while loop without max bound |
| E_CHECK_UNHANDLED | check without else clause (bypasses typed errors) |
| E_COMPENSATION_MISSING | Non-idempotent write without defer compensate |
| E_POLICY_MAY_UNGRANTED | policy_expect may X but no matching cap granted |
| E_POLICY_MUST_NOT_GRANTED | policy_expect must_not X but cap IS granted |
| E_HOLE_UNFILLED | Typed hole ?? not filled |
| W_MODEL_CONFIDENCE_HIGH_THRESHOLD | Model confidence >= 0.90 (unreliable) |

## Adding a new checker rule

1. Add the diagnostic code to `act-diagnostics/src/lib.rs` (codes module).
2. Add the check logic to `act-check/src/lib.rs`.
3. Add a test to `act-check/tests/checker_tests.rs`.

## Language design principles

- Effects, authority, uncertainty, evidence, budgets, secrets, and state are compiler-visible.
- Source code is for agents. The compiled graph IR is for runners and humans.
- Compiler diagnostics are structured JSON designed for repair by agents.
- v1 cuts: no macros, inheritance, operator overloading, reflection, unbounded loops.
