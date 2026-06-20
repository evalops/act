//! Act syntax crate: lexer and canonical AST.

pub mod ast;
pub mod lexer;

pub use ast::*;
pub use lexer::{lex, LexError, LexResult, Span as LexSpan, Token, TokenKind};
