//! Self-hosted Act programs that the runtime itself depends on.
//!
//! Today this is the accept-gate verifier ([`verify_act`]). The interpreter
//! dispatches every `infer ... accept { ... }` gate through the `verify` task
//! in `verify.act`, instead of a host primitive, so verification is subject to
//! the same budgets, effects, and traces as user code. See
//! `interp::eval_infer` for the dispatch site.

use act_parser::parse_module;
use act_syntax::ast::Module;
use std::sync::OnceLock;

/// File id used for the builtin module's spans. User modules start at 1; 0 is
/// reserved for builtin/dummy spans.
const BUILTIN_FILE: u32 = 0;

/// The parsed `verify.act` module. Parsed once on first use and cached for the
/// process lifetime; the module is immutable after parse.
pub fn verify_act() -> &'static Module {
    static MODULE: OnceLock<Module> = OnceLock::new();
    MODULE.get_or_init(|| {
        parse_module(include_str!("builtin/verify.act"), BUILTIN_FILE)
            .expect("builtin verify.act must parse")
    })
}

/// The parsed `std.act` standard-library module. Common types (Repo, Score,
/// Evidence, Hash) that programs would otherwise redeclare. Parsed once and
/// cached for the process lifetime.
pub fn std_act() -> &'static Module {
    static MODULE: OnceLock<Module> = OnceLock::new();
    MODULE.get_or_init(|| {
        parse_module(include_str!("builtin/std.act"), BUILTIN_FILE)
            .expect("builtin std.act must parse")
    })
}
