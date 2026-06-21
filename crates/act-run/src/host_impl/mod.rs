//! Concrete host implementations: a deterministic mock (for tests) and an HTTP
//! host that calls real OpenAI-compatible models and tool endpoints.

pub mod http;
pub mod mock;

pub use http::{HttpHost, OpenAiConfig};
pub use mock::{MockHost, MockInfer, MockTool};
