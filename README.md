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
  act-ir           lowering to executable graph IR
  actc             CLI
```

## License

MIT.
