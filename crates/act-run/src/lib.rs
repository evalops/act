//! Act runtime: an AST interpreter that executes tasks end-to-end against a
//! pluggable [`Host`] (real HTTP models/tools, or a mock for tests).
//!
//! See [`run_task`] for the entry point.

// Control-flow / diagnostic payloads carry `Value`s and message strings, so a
// few `Result<_, _>` Err variants are large. That is inherent to an AST
// interpreter and not worth boxing at every call site.
#![allow(clippy::result_large_err)]

pub mod budget;
pub mod builtin;
pub mod host;
pub mod host_impl;
pub mod interp;
pub mod registry;
pub mod schema;
pub mod value;

pub use host::{
    Host, HostError, InferRequest, InferResult, StateCell, ToolResult, VerifyRequest, VerifyResult,
};
pub use host_impl::{HttpHost, MockHost, OpenAiConfig};
pub use interp::{run_eval, run_task, RunConfig, RunError};
pub use registry::{FnRegistry, TypeRegistry};
pub use value::{coerce, from_literal, to_json, value_from_json, Value};

pub use act_diagnostics::codes;
