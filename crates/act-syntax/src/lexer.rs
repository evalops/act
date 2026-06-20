//! Lexer and token model for Act.
//!
//! Designed for machine consumption: tokens carry spans and are
//! round-trippable for the canonical formatter. Comments are preserved
//! as tokens (non-semantic) so formatter output matches input whitespace.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub file: u32,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn dummy() -> Span {
        Span {
            file: 0,
            start: 0,
            end: 0,
        }
    }
    pub fn len(self) -> u32 {
        self.end - self.start
    }
    pub fn union(self, other: Span) -> Span {
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl From<Span> for crate::ast::Span {
    fn from(s: Span) -> crate::ast::Span {
        crate::ast::Span {
            file: s.file,
            start: s.start,
            end: s.end,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Structural
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    DotDot,
    Arrow,
    FatArrow,
    Pipe,
    Backslash,
    Question,
    QuestionQuestion,
    At,
    Underscore,

    // Operators
    Eq,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    AmpAmp,
    PipePipe,
    Bang,
    LeEq,   // <= used for budgets
    PipeGt, // |>

    // Literals
    Int,
    Decimal,
    String,
    Duration,
    Money,
    Ident,
    ShebangLine,

    // Keywords
    KwModule,
    KwUse,
    KwAs,
    KwTool,
    KwLib,
    KwModel,
    KwVersion,
    KwType,
    KwWhere,
    KwFn,
    KwProc,
    KwTask,
    KwAgent,
    KwExtern,
    KwEffects,
    KwNeeds,
    KwBudget,
    KwCap,
    KwPolicyExpect,
    KwMay,
    KwMustNot,
    KwRequireHuman,
    KwRequire,
    KwCheck,
    KwEnsure,
    KwLet,
    KwVar,
    KwIf,
    KwElse,
    KwFor,
    KwWhile,
    KwMatch,
    KwReturn,
    KwIn,
    KwFrom,
    KwParallel,
    KwLimit,
    KwMax,
    KwOn,
    KwMessage,
    KwEvent,
    KwTry,
    KwAwait,
    KwAll,
    KwMap,
    KwRace,
    KwQuorum,
    KwOf,
    KwFirstOk,
    KwTimeout,
    KwInfer,
    KwDecide,
    KwUsing,
    KwSpawn,
    KwWith,
    KwScore,
    KwBy,
    KwRecover,
    KwDefer,
    KwCompensate,
    KwTrace,
    KwCheckpoint,
    KwInvariant,
    KwBefore,
    KwTest,
    KwEval,
    KwReplay,
    KwSandbox,
    KwOk,
    KwErr,
    KwSome,
    KwNone,
    KwTrue,
    KwFalse,
    KwNull,
    KwMd,
    KwGoal,
    KwInput,
    KwConstraints,
    KwRubric,
    KwChoices,
    KwValidate,
    KwAccept,
    KwElse_, // already have else; kept separate if needed
    KwAnd,
    KwOr,
    KwNot,
    KwEach,
    KwConfidence,
    KwValue,
    KwEvidence,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug)]
pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    file: u32,
    /// Trailing trivia (whitespace/comments) attached to the previous token.
    pending_trivia: Vec<Trivia>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trivia {
    pub span: Span,
    pub kind: TriviaKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TriviaKind {
    LineComment,
    BlockComment,
    Whitespace,
    Newline,
}

pub type LexResult = Result<Vec<Token>, LexError>;

#[derive(Clone, Debug)]
pub struct LexError {
    pub span: Span,
    pub message: String,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str, file: u32) -> Lexer<'a> {
        Lexer {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            file,
            pending_trivia: Vec::new(),
        }
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span {
            file: self.file,
            start: start as u32,
            end: end as u32,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn peek2(&self) -> Option<u8> {
        self.bytes.get(self.pos + 1).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_trivia(&mut self) {
        loop {
            let start = self.pos;
            let b = match self.peek() {
                Some(b) => b,
                None => return,
            };
            if b == b' ' || b == b'\t' || b == b'\r' {
                self.pos += 1;
                while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\r')) {
                    self.pos += 1;
                }
                self.pending_trivia.push(Trivia {
                    span: self.span(start, self.pos),
                    kind: TriviaKind::Whitespace,
                    text: self.src[start..self.pos].to_string(),
                });
                continue;
            }
            if b == b'\n' {
                self.pos += 1;
                self.pending_trivia.push(Trivia {
                    span: self.span(start, self.pos),
                    kind: TriviaKind::Newline,
                    text: "\n".to_string(),
                });
                continue;
            }
            if b == b'/' && self.peek2() == Some(b'/') {
                self.pos += 2;
                while let Some(b) = self.peek() {
                    if b == b'\n' {
                        break;
                    }
                    self.pos += 1;
                }
                self.pending_trivia.push(Trivia {
                    span: self.span(start, self.pos),
                    kind: TriviaKind::LineComment,
                    text: self.src[start..self.pos].to_string(),
                });
                continue;
            }
            if b == b'/' && self.peek2() == Some(b'*') {
                self.pos += 2;
                let mut depth = 1u32;
                while depth > 0 {
                    match self.bump() {
                        Some(b'/') if self.peek() == Some(b'*') => {
                            self.pos += 1;
                            depth += 1;
                        }
                        Some(b'*') if self.peek() == Some(b'/') => {
                            self.pos += 1;
                            depth -= 1;
                        }
                        Some(_) => {}
                        None => break,
                    }
                }
                self.pending_trivia.push(Trivia {
                    span: self.span(start, self.pos),
                    kind: TriviaKind::BlockComment,
                    text: self.src[start..self.pos].to_string(),
                });
                continue;
            }
            return;
        }
    }

    pub fn tokenize(mut self) -> LexResult {
        let mut out = Vec::new();
        loop {
            self.skip_trivia();
            // Trivia attached to following token; we don't currently store it,
            // but the lexer structure keeps that option open.
            self.pending_trivia.clear();
            let start = self.pos;
            let b = match self.peek() {
                Some(b) => b,
                None => break,
            };
            let kind = self.lex_one(b, start);
            match kind {
                Ok(Some(tok)) => out.push(tok),
                Ok(None) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }

    fn lex_one(&mut self, b: u8, start: usize) -> Result<Option<Token>, LexError> {
        // Punctuation
        macro_rules! tok {
            ($k:expr, $len:expr) => {{
                let s = self.span(start, start + $len);
                self.pos = start + $len;
                return Ok(Some(Token {
                    kind: $k,
                    span: s,
                    text: self.src[start..start + $len].to_string(),
                }));
            }};
        }
        match b {
            b'(' => tok!(TokenKind::LParen, 1),
            b')' => tok!(TokenKind::RParen, 1),
            b'{' => tok!(TokenKind::LBrace, 1),
            b'}' => tok!(TokenKind::RBrace, 1),
            b'[' => tok!(TokenKind::LBracket, 1),
            b']' => tok!(TokenKind::RBracket, 1),
            b',' => tok!(TokenKind::Comma, 1),
            b':' => {
                if self.peek2() == Some(b':') {
                    tok!(TokenKind::DotDot, 2)
                }
                tok!(TokenKind::Colon, 1)
            }
            b';' => tok!(TokenKind::Semicolon, 1),
            b'.' => tok!(TokenKind::Dot, 1),
            b'@' => tok!(TokenKind::At, 1),
            b'_' => {
                // Could be `_` wildcard or part of identifier.
                if self.is_ident_start(b) {
                    self.lex_ident(start)
                } else {
                    tok!(TokenKind::Underscore, 1)
                }
            }
            b'?' => {
                if self.peek2() == Some(b'?') {
                    tok!(TokenKind::QuestionQuestion, 2)
                }
                tok!(TokenKind::Question, 1)
            }
            b'\\' => tok!(TokenKind::Backslash, 1),
            b'+' => tok!(TokenKind::Plus, 1),
            b'-' => {
                if self.peek2() == Some(b'>') {
                    tok!(TokenKind::Arrow, 2)
                }
                tok!(TokenKind::Minus, 1)
            }
            b'*' => tok!(TokenKind::Star, 1),
            b'/' => tok!(TokenKind::Slash, 1),
            b'%' => tok!(TokenKind::Percent, 1),
            b'|' => {
                if self.peek2() == Some(b'|') {
                    tok!(TokenKind::PipePipe, 2)
                }
                if self.peek2() == Some(b'>') {
                    tok!(TokenKind::PipeGt, 2)
                }
                tok!(TokenKind::Pipe, 1)
            }
            b'&' => {
                if self.peek2() == Some(b'&') {
                    tok!(TokenKind::AmpAmp, 2)
                }
                return Err(self.err(start, "Expected `&&`"));
            }
            b'!' => {
                if self.peek2() == Some(b'=') {
                    tok!(TokenKind::BangEq, 2)
                }
                tok!(TokenKind::Bang, 1)
            }
            b'=' => {
                if self.peek2() == Some(b'=') {
                    tok!(TokenKind::EqEq, 2)
                }
                if self.peek2() == Some(b'>') {
                    tok!(TokenKind::FatArrow, 2)
                }
                tok!(TokenKind::Eq, 1)
            }
            b'<' => {
                if self.peek2() == Some(b'=') {
                    tok!(TokenKind::LeEq, 2)
                }
                tok!(TokenKind::Lt, 1)
            }
            b'>' => {
                if self.peek2() == Some(b'=') {
                    tok!(TokenKind::Ge, 2)
                }
                tok!(TokenKind::Gt, 1)
            }
            b'"' => self.lex_string(start),
            b'#' if start == 0 => self.lex_shebang(start),
            b'0'..=b'9' => self.lex_number(start),
            b'a'..=b'z' | b'A'..=b'Z' => self.lex_ident(start),
            _ => Err(self.err(start, &format!("Unexpected character {:?}", b as char))),
        }
    }

    fn err(&self, start: usize, msg: &str) -> LexError {
        LexError {
            span: self.span(start, start + 1),
            message: msg.to_string(),
        }
    }

    fn is_ident_start(&self, b: u8) -> bool {
        b.is_ascii_alphabetic() || b == b'_'
    }
    fn is_ident_cont(&self, b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    fn lex_ident(&mut self, start: usize) -> Result<Option<Token>, LexError> {
        self.pos = start + 1;
        while let Some(b) = self.peek() {
            if self.is_ident_cont(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        let kind = match keyword(text) {
            Some(k) => k,
            None => TokenKind::Ident,
        };
        Ok(Some(Token {
            kind,
            span: self.span(start, self.pos),
            text: text.to_string(),
        }))
    }

    fn lex_number(&mut self, start: usize) -> Result<Option<Token>, LexError> {
        self.pos = start;
        // integer part
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let mut is_decimal = false;
        if self.peek() == Some(b'.') && matches!(self.peek2(), Some(b'0'..=b'9')) {
            is_decimal = true;
            self.pos += 1;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() || b == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        // Suffix: duration (m/s/ms) or money (USD)
        let after = self.pos;
        // Try to read a unit suffix [a-zA-Z]+
        let mut unit_end = after;
        while let Some(b) = self.bytes.get(unit_end).copied() {
            if b.is_ascii_alphabetic() {
                unit_end += 1;
            } else {
                break;
            }
        }
        let unit = &self.src[after..unit_end];
        let text_no_underscore: String = self.src[start..after]
            .chars()
            .filter(|c| *c != '_')
            .collect();
        let span = self.span(start, unit_end);
        if !unit.is_empty() {
            // Duration or money
            if unit.eq_ignore_ascii_case("m")
                || unit.eq_ignore_ascii_case("s")
                || unit.eq_ignore_ascii_case("ms")
                || unit.eq_ignore_ascii_case("h")
            {
                self.pos = unit_end;
                return Ok(Some(Token {
                    kind: TokenKind::Duration,
                    span,
                    text: format!("{}{}", text_no_underscore, unit),
                }));
            }
            // Treat uppercase 3-letter-ish as money currency (USD, EUR, etc.)
            if unit.chars().all(|c| c.is_ascii_uppercase()) && unit.len() <= 4 {
                self.pos = unit_end;
                return Ok(Some(Token {
                    kind: TokenKind::Money,
                    span,
                    text: format!("{} {}", text_no_underscore, unit),
                }));
            }
        }
        self.pos = after;
        let kind = if is_decimal {
            TokenKind::Decimal
        } else {
            TokenKind::Int
        };
        Ok(Some(Token {
            kind,
            span: self.span(start, after),
            text: text_no_underscore,
        }))
    }

    fn lex_string(&mut self, start: usize) -> Result<Option<Token>, LexError> {
        self.pos = start + 1;
        // Support triple-quoted strings: """ ... """ (used by md"""
        // Detect md""" and plain """
        // We already consumed the first quote; check for triple.
        let triple = self.peek() == Some(b'"') && self.peek2() == Some(b'"');
        if triple {
            self.pos += 2;
        }
        let mut buf = String::new();
        loop {
            let b = match self.peek() {
                Some(b) => b,
                None => return Err(self.err(start, "Unterminated string")),
            };
            if triple {
                if b == b'"'
                    && self.peek2() == Some(b'"')
                    && self.bytes.get(self.pos + 2) == Some(&b'"')
                {
                    self.pos += 3;
                    break;
                }
                if b == b'\\' {
                    self.pos += 1;
                    if let Some(esc) = self.peek() {
                        self.pos += 1;
                        push_escaped(esc, &mut buf);
                    }
                    continue;
                }
                buf.push(b as char);
                self.pos += 1;
                continue;
            }
            if b == b'"' {
                self.pos += 1;
                break;
            }
            if b == b'\\' {
                self.pos += 1;
                if let Some(esc) = self.peek() {
                    self.pos += 1;
                    push_escaped(esc, &mut buf);
                }
                continue;
            }
            if b == b'\n' {
                return Err(self.err(start, "Newline in single-line string"));
            }
            buf.push(b as char);
            self.pos += 1;
        }
        Ok(Some(Token {
            kind: TokenKind::String,
            span: self.span(start, self.pos),
            text: buf,
        }))
    }

    fn lex_shebang(&mut self, start: usize) -> Result<Option<Token>, LexError> {
        while let Some(b) = self.peek() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
        Ok(Some(Token {
            kind: TokenKind::ShebangLine,
            span: self.span(start, self.pos),
            text: self.src[start..self.pos].to_string(),
        }))
    }
}

fn push_escaped(esc: u8, buf: &mut String) {
    match esc {
        b'n' => buf.push('\n'),
        b't' => buf.push('\t'),
        b'r' => buf.push('\r'),
        b'\\' => buf.push('\\'),
        b'"' => buf.push('"'),
        b'0' => buf.push('\0'),
        other => buf.push(other as char),
    }
}

fn keyword(s: &str) -> Option<TokenKind> {
    Some(match s {
        "module" => TokenKind::KwModule,
        "use" => TokenKind::KwUse,
        "as" => TokenKind::KwAs,
        "tool" => TokenKind::KwTool,
        "lib" => TokenKind::KwLib,
        "model" => TokenKind::KwModel,
        "type" => TokenKind::KwType,
        "where" => TokenKind::KwWhere,
        "fn" => TokenKind::KwFn,
        "proc" => TokenKind::KwProc,
        "task" => TokenKind::KwTask,
        "agent" => TokenKind::KwAgent,
        "extern" => TokenKind::KwExtern,
        "effects" => TokenKind::KwEffects,
        "needs" => TokenKind::KwNeeds,
        "budget" => TokenKind::KwBudget,
        "cap" => TokenKind::KwCap,
        "policy_expect" => TokenKind::KwPolicyExpect,
        "may" => TokenKind::KwMay,
        "must_not" => TokenKind::KwMustNot,
        "require_human" => TokenKind::KwRequireHuman,
        "require" => TokenKind::KwRequire,
        "check" => TokenKind::KwCheck,
        "ensure" => TokenKind::KwEnsure,
        "let" => TokenKind::KwLet,
        "var" => TokenKind::KwVar,
        "if" => TokenKind::KwIf,
        "else" => TokenKind::KwElse,
        "for" => TokenKind::KwFor,
        "while" => TokenKind::KwWhile,
        "match" => TokenKind::KwMatch,
        "return" => TokenKind::KwReturn,
        "in" => TokenKind::KwIn,
        "from" => TokenKind::KwFrom,
        "parallel" => TokenKind::KwParallel,
        "limit" => TokenKind::KwLimit,
        "max" => TokenKind::KwMax,
        "on" => TokenKind::KwOn,
        "message" => TokenKind::KwMessage,
        "event" => TokenKind::KwEvent,
        "try" => TokenKind::KwTry,
        "await" => TokenKind::KwAwait,
        "all" => TokenKind::KwAll,
        "map" => TokenKind::KwMap,
        "race" => TokenKind::KwRace,
        "quorum" => TokenKind::KwQuorum,
        "of" => TokenKind::KwOf,
        "first_ok" => TokenKind::KwFirstOk,
        "timeout" => TokenKind::KwTimeout,
        "infer" => TokenKind::KwInfer,
        "decide" => TokenKind::KwDecide,
        "using" => TokenKind::KwUsing,
        "spawn" => TokenKind::KwSpawn,
        "with" => TokenKind::KwWith,
        "score" => TokenKind::KwScore,
        "by" => TokenKind::KwBy,
        "recover" => TokenKind::KwRecover,
        "defer" => TokenKind::KwDefer,
        "compensate" => TokenKind::KwCompensate,
        "trace" => TokenKind::KwTrace,
        "checkpoint" => TokenKind::KwCheckpoint,
        "invariant" => TokenKind::KwInvariant,
        "before" => TokenKind::KwBefore,
        "test" => TokenKind::KwTest,
        "eval" => TokenKind::KwEval,
        "replay" => TokenKind::KwReplay,
        "sandbox" => TokenKind::KwSandbox,
        "ok" => TokenKind::KwOk,
        "err" => TokenKind::KwErr,
        "some" => TokenKind::KwSome,
        "none" => TokenKind::KwNone,
        "true" => TokenKind::KwTrue,
        "false" => TokenKind::KwFalse,
        "null" => TokenKind::KwNull,
        "md" => TokenKind::KwMd,
        "goal" => TokenKind::KwGoal,
        "input" => TokenKind::KwInput,
        "constraints" => TokenKind::KwConstraints,
        "rubric" => TokenKind::KwRubric,
        "choices" => TokenKind::KwChoices,
        "validate" => TokenKind::KwValidate,
        "accept" => TokenKind::KwAccept,
        "and" => TokenKind::KwAnd,
        "or" => TokenKind::KwOr,
        "not" => TokenKind::KwNot,
        "each" => TokenKind::KwEach,
        "confidence" => TokenKind::KwConfidence,
        "value" => TokenKind::KwValue,
        "evidence" => TokenKind::KwEvidence,
        "asc" | "desc" => return None, // treated as identifiers in score-by for now
        _ => return None,
    })
}

/// Lex a source string into tokens.
pub fn lex(src: &str, file: u32) -> LexResult {
    Lexer::new(src, file).tokenize()
}
