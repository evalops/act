//! Parser for Act. Produces a canonical AST from tokens.
//!
//! Hand-written recursive descent. Optimized for generated code:
//! explicit, boring, with good spans and recoverable errors.

use act_syntax::ast::*;
use act_syntax::lexer::{Span as LexSpan, Token, TokenKind};

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Clone, Debug)]
pub struct ParseError {
    pub span: LexSpan,
    pub message: String,
}

/// Keywords that are also acceptable as identifiers in name positions
/// (variant names, field names, etc.). This avoids forcing users to
/// escape common words like `ok`, `err`, `value`, `confidence`.
fn is_soft_keyword(k: TokenKind) -> bool {
    matches!(
        k,
        TokenKind::KwOk
            | TokenKind::KwErr
            | TokenKind::KwSome
            | TokenKind::KwNone
            | TokenKind::KwValue
            | TokenKind::KwConfidence
            | TokenKind::KwEvidence
            | TokenKind::KwInput
            | TokenKind::KwGoal
            | TokenKind::KwChoices
            | TokenKind::KwConstraints
            | TokenKind::KwRubric
            | TokenKind::KwValidate
            | TokenKind::KwAccept
            | TokenKind::KwTest
            | TokenKind::KwEval
            | TokenKind::KwModel
            | TokenKind::KwMessage
            | TokenKind::KwEvent
            | TokenKind::KwOn
            | TokenKind::KwTool
            | TokenKind::KwLib
    )
}

/// Allows span_from to accept either lexer spans or AST spans.
trait SpanLike {
    fn to_lex(self) -> LexSpan;
}
impl SpanLike for LexSpan {
    fn to_lex(self) -> LexSpan {
        self
    }
}
impl SpanLike for Span {
    fn to_lex(self) -> LexSpan {
        LexSpan {
            file: self.file,
            start: self.start,
            end: self.end,
        }
    }
}

pub struct Parser {
    toks: Vec<Token>,
    idx: usize,
    _file: u32,
}

impl Parser {
    pub fn new(toks: Vec<Token>, file: u32) -> Parser {
        Parser {
            toks,
            idx: 0,
            _file: file,
        }
    }

    fn span(&self, t: &Token) -> Span {
        Span {
            file: t.span.file,
            start: t.span.start,
            end: t.span.end,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.idx)
    }
    fn peek_kind(&self) -> Option<TokenKind> {
        self.peek().map(|t| t.kind)
    }
    fn peek2(&self) -> Option<&Token> {
        self.toks.get(self.idx + 1)
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.toks.get(self.idx).cloned();
        if t.is_some() {
            self.idx += 1;
        }
        t
    }

    fn at(&self, k: TokenKind) -> bool {
        self.peek_kind() == Some(k)
    }

    /// True if the current token starts a named label: an identifier or soft
    /// keyword immediately followed by `:`. Used for call args (`name: value`)
    /// and record fields, so soft keywords like `value` work as names.
    fn at_named_label(&self) -> bool {
        let is_name = self.at(TokenKind::Ident) || self.peek_kind().is_some_and(is_soft_keyword);
        is_name && self.peek2().map(|t| t.kind) == Some(TokenKind::Colon)
    }

    fn eat(&mut self, k: TokenKind) -> bool {
        if self.at(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, k: TokenKind, what: &str) -> ParseResult<Token> {
        match self.peek() {
            Some(t) if t.kind == k => Ok(self.bump().unwrap()),
            Some(t) => Err(ParseError {
                span: t.span,
                message: format!("Expected {}, found `{}`", what, t.text),
            }),
            None => Err(ParseError {
                span: LexSpan::dummy(),
                message: format!("Expected {}, found end of input", what),
            }),
        }
    }

    fn err(&self, span: LexSpan, msg: impl Into<String>) -> ParseError {
        ParseError {
            span,
            message: msg.into(),
        }
    }

    fn ident(&mut self) -> ParseResult<Ident> {
        let t = match self.peek_kind() {
            Some(TokenKind::Ident) => self.bump().unwrap(),
            Some(k) if is_soft_keyword(k) => self.bump().unwrap(),
            _ => {
                return Err(ParseError {
                    span: self.peek().map(|t| t.span).unwrap_or_else(LexSpan::dummy),
                    message: format!(
                        "Expected identifier, found `{}`",
                        self.peek()
                            .map(|t| t.text.as_str())
                            .unwrap_or("end of input")
                    ),
                })
            }
        };
        Ok(Ident::new(self.span(&t), t.text))
    }

    /// Parse a version string after `@`: accepts Ident, Int, or Decimal.
    fn version(&mut self) -> ParseResult<String> {
        let first = match self.peek_kind() {
            Some(TokenKind::Ident) | Some(TokenKind::Int) | Some(TokenKind::Decimal) => {
                self.bump().unwrap().text
            }
            _ => {
                return Err(self.err(
                    self.peek().map(|t| t.span).unwrap_or_else(LexSpan::dummy),
                    "Expected version identifier",
                ))
            }
        };
        let mut version = first;
        while let Some(k) = self.peek_kind() {
            match k {
                TokenKind::Minus => {
                    self.bump();
                    version.push('-');
                }
                TokenKind::Int | TokenKind::Decimal => {
                    version.push_str(&self.bump().unwrap().text);
                }
                TokenKind::Dot => {
                    self.bump();
                    version.push('.');
                }
                _ => break,
            }
        }
        Ok(version)
    }

    /// Parse a dotted path: `a.b.c`. A single ident is also a path.
    fn path(&mut self) -> ParseResult<Path> {
        let first = self.ident()?;
        let mut path = vec![first];
        while self.eat(TokenKind::Dot) {
            path.push(self.ident()?);
        }
        Ok(path)
    }

    fn span_from(&self, start: impl SpanLike, end: impl SpanLike) -> Span {
        let s = start.to_lex();
        let e = end.to_lex();
        Span {
            file: s.file,
            start: s.start,
            end: e.end,
        }
    }

    pub fn parse_module(&mut self) -> ParseResult<Module> {
        // optional shebang
        if self.at(TokenKind::ShebangLine) {
            self.bump();
        }
        let header = self.parse_header()?;
        let mut items = Vec::new();
        while self.peek().is_some() {
            items.push(self.parse_item()?);
        }
        Ok(Module {
            span: header.span,
            header,
            items,
        })
    }

    fn parse_header(&mut self) -> ParseResult<ModuleHeader> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        // `module a.b.c@0.1`
        self.expect(TokenKind::KwModule, "`module`")?;
        let name = self.path()?;
        let version = if self.eat(TokenKind::At) {
            Some(self.version()?)
        } else {
            None
        };
        let mut uses = Vec::new();
        while self.at(TokenKind::KwUse) {
            uses.push(self.parse_use()?);
        }
        let end = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        Ok(ModuleHeader {
            span: self.span_from(start, end),
            name,
            version,
            uses,
        })
    }

    fn parse_use(&mut self) -> ParseResult<UseDecl> {
        let start = self.expect(TokenKind::KwUse, "`use`")?.span;
        let kind = match self.peek_kind() {
            Some(TokenKind::KwTool) => {
                self.bump();
                UseKind::Tool
            }
            Some(TokenKind::KwLib) => {
                self.bump();
                UseKind::Lib
            }
            Some(TokenKind::KwModel) => {
                self.bump();
                UseKind::Model
            }
            _ => UseKind::Use,
        };
        let path = self.path()?;
        let version = if self.eat(TokenKind::At) {
            Some(self.version()?)
        } else {
            None
        };
        let alias = if self.eat(TokenKind::KwAs) {
            Some(self.ident()?)
        } else {
            None
        };
        Ok(UseDecl {
            span: self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start)),
            kind,
            path,
            version,
            alias,
        })
    }

    fn parse_item(&mut self) -> ParseResult<Item> {
        match self.peek_kind() {
            Some(TokenKind::KwType) => self.parse_type_decl().map(Item::TypeDecl),
            Some(TokenKind::KwFn) => self.parse_fn(FnKind::Fn).map(|d| Item::Fn(Box::new(d))),
            Some(TokenKind::KwProc) => self.parse_fn(FnKind::Proc).map(|d| Item::Proc(Box::new(d))),
            Some(TokenKind::KwTask) => self.parse_fn(FnKind::Task).map(|d| Item::Task(Box::new(d))),
            Some(TokenKind::KwAgent) => self.parse_agent().map(Item::Agent),
            Some(TokenKind::KwExtern) => self.parse_extern(),
            Some(TokenKind::KwTest) => self.parse_test(false).map(Item::Test),
            Some(TokenKind::KwEval) => self.parse_test(true).map(Item::Eval),
            Some(k) => {
                let t = self.peek().unwrap().clone();
                Err(self.err(t.span, format!("Unexpected token `{:?}` at item level", k)))
            }
            None => Err(self.err(LexSpan::dummy(), "Unexpected end of input")),
        }
    }

    fn parse_extern(&mut self) -> ParseResult<Item> {
        let start = self.expect(TokenKind::KwExtern, "`extern`")?.span;
        match self.peek_kind() {
            Some(TokenKind::KwTool) => {
                self.bump();
                self.parse_extern_tool(start)
                    .map(|t| Item::ExternTool(Box::new(t)))
            }
            Some(TokenKind::KwModel) => {
                self.bump();
                self.parse_extern_model(start)
                    .map(|t| Item::ExternModel(Box::new(t)))
            }
            _ => Err(self.err(start, "Expected `tool` or `model` after `extern`")),
        }
    }

    fn parse_extern_tool(&mut self, start: LexSpan) -> ParseResult<ExternTool> {
        let path = self.path()?;
        self.expect(TokenKind::LParen, "`(`")?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen, "`)`")?;
        self.expect(TokenKind::Arrow, "`->`")?;
        let return_ty = self.parse_ty()?;
        let mut effects = Vec::new();
        if self.eat(TokenKind::KwEffects) {
            effects = self.parse_effect_list()?;
        }
        let mut needs = Vec::new();
        if self.eat(TokenKind::KwNeeds) {
            needs = self.parse_needs()?;
        }
        let timeout = if self.eat(TokenKind::KwTimeout) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let idempotent_by = if self.at(TokenKind::Ident)
            && self.peek().map(|t| t.text.as_str()) == Some("idempotent")
        {
            self.bump();
            self.expect(TokenKind::KwBy, "`by`")?;
            Some(self.parse_expr()?)
        } else {
            None
        };
        let retry =
            if self.at(TokenKind::Ident) && self.peek().map(|t| t.text.as_str()) == Some("retry") {
                self.bump();
                self.expect(TokenKind::LBrace, "`{`")?;
                let rstart = self.peek().map(|t| t.span).unwrap_or(start);
                self.expect(TokenKind::Ident, "`attempts`")?; // attempts:
                self.expect(TokenKind::Colon, "`:`")?;
                let attempts = self.parse_expr()?;
                self.expect(TokenKind::Comma, "`,`")?;
                self.expect(TokenKind::Ident, "`on`")?;
                self.expect(TokenKind::Colon, "`:`")?;
                let mut on = Vec::new();
                loop {
                    on.push(self.path()?);
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::Comma, "`,`")?;
                self.expect(TokenKind::Ident, "`backoff`")?;
                self.expect(TokenKind::Colon, "`:`")?;
                let backoff = self.parse_expr()?;
                let end = self.expect(TokenKind::RBrace, "`}`")?.span;
                Some(RetrySpec {
                    span: self.span_from(rstart, end),
                    attempts,
                    on,
                    backoff,
                })
            } else {
                None
            };
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(ExternTool {
            span: self.span_from(start, end),
            path,
            params,
            return_ty,
            effects,
            needs,
            timeout,
            idempotent_by,
            retry,
        })
    }

    fn parse_extern_model(&mut self, start: LexSpan) -> ParseResult<ExternModel> {
        let path = self.path()?;
        let version = if self.eat(TokenKind::At) {
            Some(self.version()?)
        } else {
            None
        };
        let alias = if self.eat(TokenKind::KwAs) {
            Some(self.ident()?)
        } else {
            None
        };
        let mut full = path;
        if let Some(v) = &version {
            if let Some(last) = full.last_mut() {
                last.node = format!("{}@{}", last.node, v);
            }
        }
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(ExternModel {
            span: self.span_from(start, end),
            path: full,
            alias,
        })
    }

    fn parse_type_decl(&mut self) -> ParseResult<TypeDecl> {
        let start = self.expect(TokenKind::KwType, "`type`")?.span;
        let name = self.ident()?;
        let generics = self.parse_generics()?;
        // body forms:
        //   type T = Expr            (alias)
        //   type T = T where ...     (refinement)
        //   type T = { ... }         (record)
        //   type T = | a | b         (enum)
        //   type T                   (opaque)
        let _ = generics;
        let body = if self.eat(TokenKind::Eq) {
            if self.eat(TokenKind::Pipe) {
                // enum; first variant consumed the leading pipe
                let mut variants = Vec::new();
                loop {
                    let vstart = self.peek().map(|t| t.span).unwrap_or(start);
                    let vname = self.ident()?;
                    let fields = if self.eat(TokenKind::LParen) {
                        let mut fields = Vec::new();
                        loop {
                            let name = if self
                                .peek_kind()
                                .map(|k| k == TokenKind::Ident || is_soft_keyword(k))
                                .unwrap_or(false)
                                && self.peek2().map(|t| t.kind) == Some(TokenKind::Colon)
                            {
                                let n = self.ident()?;
                                self.expect(TokenKind::Colon, "`:`")?;
                                Some(n)
                            } else {
                                None
                            };
                            let ty = self.parse_ty()?;
                            fields.push((name, ty));
                            if !self.eat(TokenKind::Comma) {
                                break;
                            }
                        }
                        self.expect(TokenKind::RParen, "`)`")?;
                        fields
                    } else {
                        Vec::new()
                    };
                    let vend = self.peek().map(|t| t.span).unwrap_or(vstart);
                    variants.push(Variant {
                        span: self.span_from(vstart, vend),
                        name: vname,
                        fields,
                    });
                    if !self.eat(TokenKind::Pipe) {
                        break;
                    }
                }
                TypeBody::Enum(variants)
            } else if self.at(TokenKind::LBrace) {
                // record: type T = { field: T, ... }
                self.bump();
                TypeBody::Record(self.parse_record_fields()?)
            } else {
                let ty = self.parse_ty()?;
                if self.eat(TokenKind::KwWhere) {
                    let mut preds = Vec::new();
                    loop {
                        preds.push(self.parse_expr()?);
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    TypeBody::Refinement {
                        ty: Box::new(ty),
                        predicates: preds,
                    }
                } else {
                    TypeBody::Alias(Box::new(ty))
                }
            }
        } else if self.eat(TokenKind::LBrace) {
            TypeBody::Record(self.parse_record_fields()?)
        } else {
            TypeBody::Opaque
        };
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(TypeDecl {
            span: self.span_from(start, end),
            name,
            kind: TypeDeclKind::Type,
            body,
        })
    }

    fn parse_record_fields(&mut self) -> ParseResult<Vec<Field>> {
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let fstart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            let name = self.ident()?;
            let optional = self.eat(TokenKind::Question);
            self.expect(TokenKind::Colon, "`:`")?;
            let ty = self.parse_ty()?;
            let fend = self.peek().map(|t| t.span).unwrap_or(fstart);
            fields.push(Field {
                span: self.span_from(fstart, fend),
                name,
                optional,
                ty,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(fields)
    }

    fn parse_fn(&mut self, kind: FnKind) -> ParseResult<FnDecl> {
        let start = match kind {
            FnKind::Fn => self.expect(TokenKind::KwFn, "`fn`")?.span,
            FnKind::Proc => self.expect(TokenKind::KwProc, "`proc`")?.span,
            FnKind::Task => self.expect(TokenKind::KwTask, "`task`")?.span,
        };
        let name = self.ident()?;
        let generics = self.parse_generics()?;
        self.expect(TokenKind::LParen, "`(`")?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen, "`)`")?;
        self.expect(TokenKind::Arrow, "`->`")?;
        let return_ty = self.parse_ty()?;
        let mut effects = Vec::new();
        if self.eat(TokenKind::KwEffects) {
            effects = self.parse_effect_list()?;
        }
        let mut needs = Vec::new();
        if self.eat(TokenKind::KwNeeds) {
            needs = self.parse_needs()?;
        }
        let budget = if self.eat(TokenKind::KwBudget) {
            Some(self.parse_budget()?)
        } else {
            None
        };
        let policy_expect = if self.eat(TokenKind::KwPolicyExpect) {
            Some(self.parse_policy_expect()?)
        } else {
            None
        };
        let body = if self.eat(TokenKind::LBrace) {
            let b = self.parse_block()?;
            self.expect(TokenKind::RBrace, "`}`")?;
            Some(b)
        } else {
            None
        };
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(FnDecl {
            span: self.span_from(start, end),
            kind,
            name,
            generics,
            params,
            return_ty,
            effects,
            needs,
            budget,
            policy_expect,
            body,
        })
    }

    fn parse_generics(&mut self) -> ParseResult<Vec<Ident>> {
        let mut generics = Vec::new();
        if self.eat(TokenKind::Lt) {
            loop {
                generics.push(self.ident()?);
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::Gt, "`>`")?;
        }
        Ok(generics)
    }

    fn parse_params(&mut self) -> ParseResult<Vec<Param>> {
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) && self.peek().is_some() {
            let pstart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            let is_cap = self.eat(TokenKind::KwCap);
            let name = self.ident()?;
            self.expect(TokenKind::Colon, "`:`")?;
            let ty = self.parse_ty()?;
            let default = if self.eat(TokenKind::Eq) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            let pend = self.peek().map(|t| t.span).unwrap_or(pstart);
            params.push(Param {
                span: self.span_from(pstart, pend),
                name,
                is_cap,
                ty,
                default,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(params)
    }

    fn parse_effect_list(&mut self) -> ParseResult<Vec<Spanned<EffectRef>>> {
        self.expect(TokenKind::LBracket, "`[`")?;
        let mut effects = Vec::new();
        while !self.at(TokenKind::RBracket) {
            let estart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            let path = self.path()?;
            let eend = self.peek().map(|t| t.span).unwrap_or(estart);
            effects.push(Spanned::new(
                self.span_from(estart, eend),
                EffectRef { path },
            ));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket, "`]`")?;
        Ok(effects)
    }

    fn parse_needs(&mut self) -> ParseResult<Vec<Spanned<Capability>>> {
        self.expect(TokenKind::LBracket, "`[`")?;
        let mut caps = Vec::new();
        while !self.at(TokenKind::RBracket) {
            let cstart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            self.expect(TokenKind::KwCap, "`cap`")?;
            let path = self.path()?;
            let mut args = Vec::new();
            if self.eat(TokenKind::LParen) {
                while !self.at(TokenKind::RParen) {
                    args.push(self.parse_expr()?);
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "`)`")?;
            }
            let cend = self.peek().map(|t| t.span).unwrap_or(cstart);
            caps.push(Spanned::new(
                self.span_from(cstart, cend),
                Capability { path, args },
            ));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket, "`]`")?;
        Ok(caps)
    }

    fn parse_budget(&mut self) -> ParseResult<Budget> {
        let start = self.expect(TokenKind::LBrace, "`{`")?.span;
        let mut limits = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let lstart = self.peek().map(|t| t.span).unwrap_or(start);
            let metric = match self.peek_kind() {
                Some(TokenKind::Ident) => {
                    let t = self.bump().unwrap();
                    match t.text.as_str() {
                        "wall_time" => BudgetMetric::WallTime,
                        "tokens" => BudgetMetric::Tokens,
                        "cost" => BudgetMetric::Cost,
                        "tool_calls" => BudgetMetric::ToolCalls,
                        other => {
                            return Err(
                                self.err(t.span, format!("Unknown budget metric `{}`", other))
                            )
                        }
                    }
                }
                _ => return Err(self.err(lstart, "Expected budget metric")),
            };
            let op = match self.peek_kind() {
                Some(TokenKind::LeEq) => {
                    self.bump();
                    BudgetOp::Le
                }
                Some(TokenKind::Lt) => {
                    self.bump();
                    BudgetOp::Lt
                }
                Some(TokenKind::Gt) => {
                    self.bump();
                    BudgetOp::Gt
                }
                Some(TokenKind::Ge) => {
                    self.bump();
                    BudgetOp::Ge
                }
                Some(TokenKind::EqEq) => {
                    self.bump();
                    BudgetOp::Eq
                }
                _ => return Err(self.err(lstart, "Expected budget operator")),
            };
            let value = self.parse_expr()?;
            let lend = self.peek().map(|t| t.span).unwrap_or(lstart);
            limits.push(BudgetLimit {
                span: self.span_from(lstart, lend),
                metric,
                op,
                value,
            });
            if !self.eat(TokenKind::Comma) && !self.at(TokenKind::RBrace) {
                break;
            }
        }
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        Ok(Budget {
            span: self.span_from(start, end),
            limits,
        })
    }

    fn parse_policy_expect(&mut self) -> ParseResult<Spanned<PolicyExpect>> {
        let start = self.expect(TokenKind::LBrace, "`{`")?.span;
        let mut clauses = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let cstart = self.peek().map(|t| t.span).unwrap_or(start);
            let verb = match self.peek_kind() {
                Some(TokenKind::KwMay) => {
                    self.bump();
                    PolicyVerb::May
                }
                Some(TokenKind::KwMustNot) => {
                    self.bump();
                    PolicyVerb::MustNot
                }
                Some(TokenKind::KwRequireHuman) => {
                    self.bump();
                    PolicyVerb::RequireHuman
                }
                _ => return Err(self.err(cstart, "Expected `may`/`must_not`/`require_human`")),
            };
            let (target, where_clause) = match verb {
                PolicyVerb::RequireHuman => {
                    // require_human when <expr> — no target path, condition after `when`.
                    let cond = if self.at(TokenKind::Ident)
                        && self.peek().map(|t| t.text.as_str()) == Some("when")
                    {
                        self.bump();
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    (Vec::new(), cond)
                }
                PolicyVerb::May | PolicyVerb::MustNot => {
                    let target = self.path()?;
                    let where_clause = if self.eat(TokenKind::KwWhere) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    (target, where_clause)
                }
            };
            let cend = self.peek().map(|t| t.span).unwrap_or(cstart);
            clauses.push(PolicyClause {
                span: self.span_from(cstart, cend),
                verb,
                target,
                where_clause,
            });
        }
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        Ok(Spanned::new(
            self.span_from(start, end),
            PolicyExpect {
                span: self.span_from(start, end),
                clauses,
            },
        ))
    }

    fn parse_agent(&mut self) -> ParseResult<AgentDecl> {
        let start = self.expect(TokenKind::KwAgent, "`agent`")?.span;
        let name = self.ident()?;
        let state_ty =
            if self.at(TokenKind::Ident) && self.peek().map(|t| t.text.as_str()) == Some("state") {
                self.bump();
                Some(self.parse_ty()?)
            } else {
                None
            };
        let mut effects = Vec::new();
        if self.eat(TokenKind::KwEffects) {
            effects = self.parse_effect_list()?;
        }
        let mut needs = Vec::new();
        if self.eat(TokenKind::KwNeeds) {
            needs = self.parse_needs()?;
        }
        let budget = if self.eat(TokenKind::KwBudget) {
            Some(self.parse_budget()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let mut handlers = Vec::new();
        while !self.at(TokenKind::RBrace) {
            handlers.push(self.parse_event_handler()?);
        }
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        Ok(AgentDecl {
            span: self.span_from(start, end),
            name,
            state_ty,
            effects,
            needs,
            budget,
            handlers,
        })
    }

    fn parse_event_handler(&mut self) -> ParseResult<EventHandler> {
        let start = self.expect(TokenKind::KwOn, "`on`")?.span;
        let kind = if self.eat(TokenKind::KwMessage) {
            EventKind::OnMessage
        } else {
            self.eat(TokenKind::KwEvent);
            EventKind::On
        };
        let path = self.path()?;
        let trigger_span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
        self.expect(TokenKind::KwAs, "`as`")?;
        let binder = self.ident()?;
        let where_clause = if self.eat(TokenKind::KwWhere) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let budget = if self.eat(TokenKind::KwBudget) {
            Some(self.parse_budget()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        Ok(EventHandler {
            span: self.span_from(start, end),
            trigger: EventTrigger {
                span: trigger_span,
                kind,
                path,
            },
            binder,
            where_clause,
            budget,
            body,
        })
    }

    fn parse_test(&mut self, _eval: bool) -> ParseResult<TestBlock> {
        let start = self.bump().unwrap().span; // test or eval
        let label = self.ident()?;
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        Ok(TestBlock {
            span: self.span_from(start, end),
            label,
            body,
        })
    }

    fn parse_block(&mut self) -> ParseResult<Block> {
        // assumes `{` already consumed
        let mut stmts = Vec::new();
        while !self.at(TokenKind::RBrace) && self.peek().is_some() {
            let s = self.parse_stmt()?;
            stmts.push(s);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> ParseResult<Spanned<Stmt>> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        let stmt = match self.peek_kind() {
            Some(TokenKind::KwLet) => {
                self.bump();
                let mutable = false;
                let name = self.ident()?;
                let ty = if self.eat(TokenKind::Colon) {
                    Some(self.parse_ty()?)
                } else {
                    None
                };
                self.expect(TokenKind::Eq, "`=`")?;
                let init = self.parse_expr()?;
                Stmt::Let {
                    mutable,
                    name,
                    ty,
                    init,
                }
            }
            Some(TokenKind::KwVar) => {
                self.bump();
                let name = self.ident()?;
                let ty = if self.eat(TokenKind::Colon) {
                    Some(self.parse_ty()?)
                } else {
                    None
                };
                self.expect(TokenKind::Eq, "`=`")?;
                let init = self.parse_expr()?;
                Stmt::Var { name, ty, init }
            }
            Some(TokenKind::KwReturn) => {
                self.bump();
                let value = if self.at(TokenKind::RBrace) || self.at(TokenKind::Semicolon) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                Stmt::Return(value)
            }
            Some(TokenKind::KwIf) => self.parse_if()?,
            Some(TokenKind::KwFor) => self.parse_for()?,
            Some(TokenKind::KwWhile) => self.parse_while()?,
            Some(TokenKind::KwMatch) => self.parse_match()?,
            Some(TokenKind::KwRequire) => {
                self.bump();
                Stmt::Require(self.parse_expr()?)
            }
            Some(TokenKind::KwCheck) => {
                self.bump();
                let cond = self.parse_expr()?;
                let else_block = if self.eat(TokenKind::KwElse) {
                    self.expect(TokenKind::LBrace, "`{`")?;
                    let b = self.parse_block()?;
                    self.expect(TokenKind::RBrace, "`}`")?;
                    Some(b)
                } else {
                    None
                };
                Stmt::Check { cond, else_block }
            }
            Some(TokenKind::KwEnsure) => {
                self.bump();
                Stmt::Ensure(self.parse_expr()?)
            }
            Some(TokenKind::KwTrace) => self.parse_trace()?,
            Some(TokenKind::KwCheckpoint) => self.parse_checkpoint()?,
            Some(TokenKind::KwInvariant) => self.parse_invariant()?,
            Some(TokenKind::KwRecover) => self.parse_recover()?,
            Some(TokenKind::KwDefer) => self.parse_defer()?,
            _ => {
                // Could be assignment or expression statement.
                let expr = self.parse_expr()?;
                if self.eat(TokenKind::Eq) {
                    let value = self.parse_expr()?;
                    Stmt::Assign {
                        target: expr,
                        value,
                    }
                } else {
                    Stmt::Expr(expr)
                }
            }
        };
        self.eat(TokenKind::Semicolon);
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(Spanned::new(self.span_from(start, end), stmt))
    }

    fn parse_if(&mut self) -> ParseResult<Stmt> {
        self.bump(); // if
        let cond = self.parse_expr()?;
        self.expect(TokenKind::LBrace, "`{`")?;
        let then = self.parse_block()?;
        self.expect(TokenKind::RBrace, "`}`")?;
        let else_ = if self.eat(TokenKind::KwElse) {
            if self.at(TokenKind::KwIf) {
                let s = self.parse_stmt()?;
                self.expect(TokenKind::LBrace, "`{`").ok();
                // Treat nested if as a single-statement block.
                Some(vec![s])
            } else {
                self.expect(TokenKind::LBrace, "`{`")?;
                let block = self.parse_block()?;
                self.expect(TokenKind::RBrace, "`}`")?;
                Some(block)
            }
        } else {
            None
        };
        Ok(Stmt::If { cond, then, else_ })
    }

    fn parse_for(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let item = self.ident()?;
        self.expect(TokenKind::KwIn, "`in`")?;
        let iter = self.parse_expr()?;
        let limit = if self.eat(TokenKind::KwLimit) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(Stmt::For {
            item,
            iter,
            limit,
            body,
        })
    }

    fn parse_while(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let cond = self.parse_expr()?;
        let max = if self.eat(TokenKind::KwMax) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(Stmt::While { cond, max, body })
    }

    fn parse_match(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let scrutinee = self.parse_expr()?;
        self.expect(TokenKind::LBrace, "`{`")?;
        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let astart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            let pattern = self.parse_pattern()?;
            let guard = if self.eat(TokenKind::KwWhere) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            // arm body uses FatArrow => ...
            self.expect(TokenKind::FatArrow, "`=>`")?;
            // Body can be a single expr-stmt or a block. Prefer block.
            let body = if self.eat(TokenKind::LBrace) {
                let b = self.parse_block()?;
                self.expect(TokenKind::RBrace, "`}`")?;
                b
            } else {
                let s = self.parse_stmt()?;
                vec![s]
            };
            let aend = self.peek().map(|t| t.span).unwrap_or(astart);
            arms.push(MatchArm {
                span: self.span_from(astart, aend),
                pattern,
                guard,
                body,
            });
        }
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(Stmt::Match { scrutinee, arms })
    }

    fn parse_pattern(&mut self) -> ParseResult<Spanned<Pattern>> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        let pat = if self.eat(TokenKind::Underscore) {
            Pattern::Wildcard
        } else if self.at(TokenKind::KwSome)
            || self.at(TokenKind::KwNone)
            || self.at(TokenKind::KwOk)
            || self.at(TokenKind::KwErr)
        {
            let name_tok = self.bump().unwrap();
            let name_str = name_tok.text.clone();
            let binder = if self.eat(TokenKind::LParen) {
                let i = self.ident()?;
                self.expect(TokenKind::RParen, "`)`")?;
                Some(i)
            } else {
                None
            };
            let path = vec![Ident::new(self.span(&name_tok), name_str)];
            Pattern::Tag { name: path, binder }
        } else if self.at(TokenKind::Ident) {
            // Could be a variant path `Foo.Bar(x)` or a binder `x`.
            let path = self.path()?;
            if self.eat(TokenKind::LParen) {
                let binder = self.ident()?;
                self.expect(TokenKind::RParen, "`)`")?;
                Pattern::Tag {
                    name: path,
                    binder: Some(binder),
                }
            } else if path.len() == 1 {
                Pattern::Bind(path.into_iter().next().unwrap())
            } else {
                Pattern::Tag {
                    name: path,
                    binder: None,
                }
            }
        } else {
            let lit = self.parse_expr()?;
            Pattern::Lit(lit)
        };
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(Spanned::new(self.span_from(start, end), pat))
    }

    fn parse_trace(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let label = if self.at(TokenKind::String) {
            let t = self.bump().unwrap();
            Ident::new(self.span(&t), t.text)
        } else {
            self.ident()?
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let name = self.ident()?;
            self.expect(TokenKind::Colon, "`:`")?;
            let value = self.parse_expr()?;
            fields.push((name, value));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(Stmt::Trace { label, fields })
    }

    fn parse_checkpoint(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let label = self.ident()?;
        // `score <expr> on <expr> require <expr>`
        // Simplified: body expr then optional require
        let body = self.parse_expr()?;
        let require = if self.eat(TokenKind::KwRequire) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Stmt::Checkpoint {
            label,
            body,
            require,
        })
    }

    fn parse_invariant(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let label = self.ident()?;
        let before = if self.eat(TokenKind::KwBefore) {
            Some(self.path()?)
        } else {
            None
        };
        self.expect(TokenKind::KwRequire, "`require`")?;
        let require = self.parse_expr()?;
        Ok(Stmt::Invariant {
            label,
            before,
            require,
        })
    }

    fn parse_recover(&mut self) -> ParseResult<Stmt> {
        self.bump();
        let error_ty = self.path()?;
        self.expect(TokenKind::KwFrom, "`from`")?;
        let from = self.parse_expr()?;
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        self.expect(TokenKind::RBrace, "`}`")?;
        let es = error_ty
            .iter()
            .last()
            .map(|i| i.span)
            .unwrap_or_else(Span::dummy);
        let span = self.span_from(es, es);
        Ok(Stmt::Recover {
            error_ty: Spanned::new(span, error_ty),
            from,
            body,
        })
    }

    fn parse_defer(&mut self) -> ParseResult<Stmt> {
        self.bump();
        // `defer compensate { ... }`
        let kind = if self.eat(TokenKind::KwCompensate) {
            DeferKind::Compensate
        } else {
            return Err(self.err(
                self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy()),
                "Expected `compensate`",
            ));
        };
        self.expect(TokenKind::LBrace, "`{`")?;
        let body = self.parse_block()?;
        self.expect(TokenKind::RBrace, "`}`")?;
        Ok(Stmt::Defer { kind, body })
    }

    // ---- Types ----

    fn parse_ty(&mut self) -> ParseResult<Spanned<Ty>> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        let ty = if self.eat(TokenKind::QuestionQuestion) {
            Ty::Hole
        } else if self.eat(TokenKind::LBracket) {
            let inner = self.parse_ty()?;
            self.expect(TokenKind::RBracket, "`]`")?;
            Ty::Array(Box::new(inner))
        } else if self.eat(TokenKind::LParen) {
            // tuple or parenthesized
            let first = self.parse_ty()?;
            if self.eat(TokenKind::Comma) {
                let mut elems = vec![first];
                while !self.at(TokenKind::RParen) {
                    elems.push(self.parse_ty()?);
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RParen, "`)`")?;
                let end = self.peek().map(|t| t.span).unwrap_or(start);
                return Ok(Spanned::new(self.span_from(start, end), Ty::Tuple(elems)));
            }
            self.expect(TokenKind::RParen, "`)`")?;
            first.node
        } else {
            let path = self.path()?;
            let mut args = Vec::new();
            if self.eat(TokenKind::Lt) {
                args.push(self.parse_ty()?);
                while self.eat(TokenKind::Comma) {
                    args.push(self.parse_ty()?);
                }
                self.expect(TokenKind::Gt, "`>`")?;
            }
            Ty::Named { path, args }
        };
        // Handle Map<K,V> and Set<T> sugar is handled via Named; fine.
        let end = self.peek().map(|t| t.span).unwrap_or(start);
        Ok(Spanned::new(self.span_from(start, end), ty))
    }

    // ---- Expressions ----

    fn parse_expr(&mut self) -> ParseResult<Spanned<Expr>> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut lhs = self.parse_and()?;
        while self.eat(TokenKind::PipePipe) {
            let rhs = self.parse_and()?;
            let span = lhs.span.union(rhs.span);
            lhs = Spanned::new(
                span,
                Expr::Bin {
                    op: BinOp::Or,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            );
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut lhs = self.parse_cmp()?;
        while self.eat(TokenKind::AmpAmp) {
            let rhs = self.parse_cmp()?;
            let span = lhs.span.union(rhs.span);
            lhs = Spanned::new(
                span,
                Expr::Bin {
                    op: BinOp::And,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            );
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut lhs = self.parse_add()?;
        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::EqEq) => BinOp::Eq,
                Some(TokenKind::BangEq) => BinOp::Ne,
                Some(TokenKind::Lt) => BinOp::Lt,
                Some(TokenKind::LeEq) => BinOp::Le,
                Some(TokenKind::Gt) => BinOp::Gt,
                Some(TokenKind::Ge) => BinOp::Ge,
                Some(TokenKind::KwIn) => {
                    self.bump();
                    let rhs = self.parse_add()?;
                    let span = lhs.span.union(rhs.span);
                    lhs = Spanned::new(
                        span,
                        Expr::Bin {
                            op: BinOp::In,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        },
                    );
                    continue;
                }
                _ => break,
            };
            self.bump();
            let rhs = self.parse_add()?;
            let span = lhs.span.union(rhs.span);
            lhs = Spanned::new(
                span,
                Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            );
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Plus) => BinOp::Add,
                Some(TokenKind::Minus) => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_mul()?;
            let span = lhs.span.union(rhs.span);
            lhs = Spanned::new(
                span,
                Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            );
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Star) => BinOp::Mul,
                Some(TokenKind::Slash) => BinOp::Div,
                Some(TokenKind::Percent) => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_unary()?;
            let span = lhs.span.union(rhs.span);
            lhs = Spanned::new(
                span,
                Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            );
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> ParseResult<Spanned<Expr>> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        match self.peek_kind() {
            Some(TokenKind::Bang) => {
                self.bump();
                let e = self.parse_unary()?;
                let span = self.span_from(start, e.span);
                Ok(Spanned::new(
                    span,
                    Expr::Un {
                        op: UnOp::Not,
                        expr: Box::new(e),
                    },
                ))
            }
            Some(TokenKind::Minus) => {
                self.bump();
                let e = self.parse_unary()?;
                let span = self.span_from(start, e.span);
                Ok(Spanned::new(
                    span,
                    Expr::Un {
                        op: UnOp::Neg,
                        expr: Box::new(e),
                    },
                ))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> ParseResult<Spanned<Expr>> {
        let mut e = self.parse_primary()?;
        loop {
            if self.eat(TokenKind::Dot) {
                let name = self.ident()?;
                if self.eat(TokenKind::LParen) {
                    let args = self.parse_call_args()?;
                    self.expect(TokenKind::RParen, "`)`")?;
                    let span = e.span.union(name.span);
                    e = Spanned::new(
                        span,
                        Expr::Method {
                            receiver: Box::new(e),
                            name,
                            args,
                        },
                    );
                } else {
                    let span = e.span.union(name.span);
                    e = Spanned::new(
                        span,
                        Expr::Field {
                            receiver: Box::new(e),
                            name,
                        },
                    );
                }
            } else if self.eat(TokenKind::LBracket) {
                let index = self.parse_expr()?;
                self.expect(TokenKind::RBracket, "`]`")?;
                let span = e.span.union(index.span);
                e = Spanned::new(
                    span,
                    Expr::Index {
                        receiver: Box::new(e),
                        index: Box::new(index),
                    },
                );
            } else if self.eat(TokenKind::LParen) {
                // call with named/positional args
                let args = self.parse_call_args()?;
                self.expect(TokenKind::RParen, "`)`")?;
                let span = e.span;
                e = Spanned::new(
                    span,
                    Expr::Call {
                        callee: Box::new(e),
                        args,
                    },
                );
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn parse_call_args(&mut self) -> ParseResult<Vec<CallArg>> {
        let mut args = Vec::new();
        while !self.at(TokenKind::RParen) && self.peek().is_some() {
            let astart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            // named arg? `name: value`
            let name = if self.at_named_label() {
                Some(self.ident()?)
            } else {
                None
            };
            if name.is_some() {
                self.expect(TokenKind::Colon, "`:`")?;
            }
            let value = self.parse_expr()?;
            let aend = self.peek().map(|t| t.span).unwrap_or(astart);
            args.push(CallArg {
                span: self.span_from(astart, aend),
                name,
                value,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> ParseResult<Spanned<Expr>> {
        let start = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
        match self.peek_kind() {
            Some(TokenKind::LParen) => {
                // Parenthesized grouping: (expr)
                self.bump();
                let inner = self.parse_expr()?;
                self.expect(TokenKind::RParen, "`)`")?;
                Ok(inner)
            }
            Some(TokenKind::Int) => {
                let t = self.bump().unwrap();
                let n: i64 = t.text.parse().unwrap_or(0);
                Ok(Spanned::new(self.span(&t), Expr::Lit(Literal::Int(n))))
            }
            Some(TokenKind::Decimal) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(
                    self.span(&t),
                    Expr::Lit(Literal::Decimal(t.text)),
                ))
            }
            Some(TokenKind::String) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(
                    self.span(&t),
                    Expr::Lit(Literal::String(t.text)),
                ))
            }
            Some(TokenKind::Duration) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(
                    self.span(&t),
                    Expr::Lit(Literal::Duration(t.text)),
                ))
            }
            Some(TokenKind::Money) => {
                let t = self.bump().unwrap();
                let parts: Vec<&str> = t.text.split_whitespace().collect();
                let amt = parts.first().map(|s| s.to_string()).unwrap_or_default();
                let cur = parts.get(1).map(|s| s.to_string()).unwrap_or_default();
                Ok(Spanned::new(
                    self.span(&t),
                    Expr::Lit(Literal::Money(amt, cur)),
                ))
            }
            Some(TokenKind::KwTrue) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(self.span(&t), Expr::Lit(Literal::Bool(true))))
            }
            Some(TokenKind::KwFalse) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(self.span(&t), Expr::Lit(Literal::Bool(false))))
            }
            Some(TokenKind::KwNull) => {
                let t = self.bump().unwrap();
                Ok(Spanned::new(self.span(&t), Expr::Lit(Literal::Null)))
            }
            Some(TokenKind::KwOk) => {
                self.bump();
                let value = if self.eat(TokenKind::LParen) {
                    let v = self.parse_expr()?;
                    self.expect(TokenKind::RParen, "`)`")?;
                    Some(Box::new(v))
                } else {
                    None
                };
                let end = self.peek().map(|t| t.span).unwrap_or(start);
                Ok(Spanned::new(
                    self.span_from(start, end),
                    Expr::ResultCtor {
                        variant: ResultVariant::Ok,
                        value,
                    },
                ))
            }
            Some(TokenKind::KwErr) => {
                self.bump();
                let value = if self.eat(TokenKind::LParen) {
                    let v = self.parse_expr()?;
                    self.expect(TokenKind::RParen, "`)`")?;
                    Some(Box::new(v))
                } else {
                    None
                };
                let end = self.peek().map(|t| t.span).unwrap_or(start);
                Ok(Spanned::new(
                    self.span_from(start, end),
                    Expr::ResultCtor {
                        variant: ResultVariant::Err,
                        value,
                    },
                ))
            }
            Some(TokenKind::KwTry) => {
                self.bump();
                let e = self.parse_unary()?;
                let span = self.span_from(start, e.span);
                Ok(Spanned::new(span, Expr::Try(Box::new(e))))
            }
            Some(TokenKind::KwAwait) => self.parse_await(start),
            Some(TokenKind::KwInfer) => self.parse_infer(start),
            Some(TokenKind::KwDecide) => self.parse_decide(start),
            Some(TokenKind::KwSpawn) => self.parse_spawn(start),
            Some(TokenKind::QuestionQuestion) => {
                self.bump();
                let hole = if self.eat(TokenKind::LBrace) {
                    let mut goal = None;
                    let mut must_satisfy = Vec::new();
                    while !self.at(TokenKind::RBrace) {
                        let name = self.ident()?;
                        self.expect(TokenKind::Colon, "`:`")?;
                        match name.node.as_str() {
                            "goal" => {
                                goal = Some(Box::new(self.parse_expr()?));
                            }
                            "must_satisfy" => {
                                self.expect(TokenKind::LBracket, "`[`")?;
                                while !self.at(TokenKind::RBracket) {
                                    must_satisfy.push(self.parse_expr()?);
                                    if !self.eat(TokenKind::Comma) {
                                        break;
                                    }
                                }
                                self.expect(TokenKind::RBracket, "`]`")?;
                            }
                            _ => {
                                self.parse_expr()?;
                            }
                        }
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBrace, "`}`")?;
                    HoleSpec::Constrained { goal, must_satisfy }
                } else {
                    let hint = self.parse_primary()?;
                    HoleSpec::Plain(Box::new(hint))
                };
                let end = self.peek().map(|t| t.span).unwrap_or(start);
                Ok(Spanned::new(self.span_from(start, end), Expr::Hole(hole)))
            }
            Some(TokenKind::KwMd) => {
                self.bump();
                // expect triple-quoted string
                let t = self.expect(TokenKind::String, "markdown string")?;
                // The lexer already consumed `md` then `"..."`. For triple-quoted we need to
                // handle interpolation later; treat as plain string for now.
                Ok(Spanned::new(
                    self.span(&t),
                    Expr::Markdown(vec![InterpPart::Str(t.text)]),
                ))
            }
            Some(TokenKind::LBrace) => {
                // Record literal `{ a: 1, b: 2 }` or block? Use record if first token is ident+colon.
                self.bump();
                if self.at(TokenKind::RBrace) {
                    self.bump();
                    let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
                    return Ok(Spanned::new(span, Expr::Record(Vec::new())));
                }
                // peek for ident:
                if self.at_named_label() {
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) {
                        let name = self.ident()?;
                        self.expect(TokenKind::Colon, "`:`")?;
                        let v = self.parse_expr()?;
                        fields.push((name, v));
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    let end = self.expect(TokenKind::RBrace, "`}`")?.span;
                    return Ok(Spanned::new(
                        self.span_from(start, end),
                        Expr::Record(fields),
                    ));
                }
                // otherwise treat as block expression
                let body = self.parse_block()?;
                let end = self.expect(TokenKind::RBrace, "`}`")?.span;
                Ok(Spanned::new(self.span_from(start, end), Expr::Block(body)))
            }
            Some(TokenKind::LBracket) => {
                self.bump();
                let mut elems = Vec::new();
                while !self.at(TokenKind::RBracket) {
                    elems.push(self.parse_expr()?);
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                let end = self.expect(TokenKind::RBracket, "`]`")?.span;
                Ok(Spanned::new(self.span_from(start, end), Expr::Array(elems)))
            }
            Some(k) if k == TokenKind::Ident || is_soft_keyword(k) => {
                let path = self.path()?;
                let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
                Ok(Spanned::new(span, Expr::Path(path)))
            }
            Some(k) => {
                let t = self.peek().unwrap().clone();
                Err(self.err(t.span, format!("Unexpected token `{:?}` in expression", k)))
            }
            None => Err(self.err(start, "Unexpected end of input in expression")),
        }
    }

    fn parse_await(&mut self, start: LexSpan) -> ParseResult<Spanned<Expr>> {
        self.bump(); // await
        let kind = match self.peek_kind() {
            Some(TokenKind::KwAll) => {
                self.bump();
                AwaitKind::All
            }
            Some(TokenKind::KwMap) => {
                self.bump();
                AwaitKind::Map
            }
            Some(TokenKind::KwRace) => {
                self.bump();
                AwaitKind::Race
            }
            Some(TokenKind::KwQuorum) => {
                self.bump();
                AwaitKind::Quorum
            }
            _ => {
                return Err(self.err(
                    self.peek().map(|t| t.span).unwrap_or(start),
                    "Expected `all`/`map`/`race`/`quorum`",
                ))
            }
        };
        let body = match kind {
            AwaitKind::All => {
                self.expect(TokenKind::LBrace, "`{`")?;
                let mut branches = Vec::new();
                while !self.at(TokenKind::RBrace) {
                    let name = self.ident()?;
                    self.expect(TokenKind::Colon, "`:`")?;
                    let v = self.parse_expr()?;
                    branches.push((name, v));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                let _end = self.expect(TokenKind::RBrace, "`}`")?.span;
                AwaitBody::All(branches)
            }
            AwaitKind::Map => {
                let item = self.ident()?;
                self.expect(TokenKind::KwIn, "`in`")?;
                let iter = self.parse_expr()?;
                let parallel = if self.eat(TokenKind::KwParallel) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                let limit = if self.eat(TokenKind::KwLimit) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                self.expect(TokenKind::LBrace, "`{`")?;
                let body = self.parse_block()?;
                self.expect(TokenKind::RBrace, "`}`")?;
                AwaitBody::Map {
                    item,
                    iter,
                    parallel,
                    limit,
                    body,
                }
            }
            AwaitKind::Race => {
                self.expect(TokenKind::KwFirstOk, "`first_ok`")?;
                self.expect(TokenKind::LBrace, "`{`")?;
                let mut branches = Vec::new();
                while !self.at(TokenKind::RBrace) {
                    let name = self.ident()?;
                    self.expect(TokenKind::Colon, "`:`")?;
                    let v = self.parse_expr()?;
                    branches.push((name, v));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace, "`}`")?;
                let timeout = if self.eat(TokenKind::KwTimeout) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                AwaitBody::Race { branches, timeout }
            }
            AwaitKind::Quorum => {
                let quorum = self.parse_expr()?;
                self.expect(TokenKind::KwOf, "`of`")?;
                let of = self.parse_expr()?;
                self.expect(TokenKind::LBrace, "`{`")?;
                let mut branches = Vec::new();
                while !self.at(TokenKind::RBrace) {
                    let name = self.ident()?;
                    self.expect(TokenKind::Colon, "`:`")?;
                    let v = self.parse_expr()?;
                    branches.push((name, v));
                    if !self.eat(TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(TokenKind::RBrace, "`}`")?;
                AwaitBody::Quorum {
                    quorum,
                    of,
                    branches,
                }
            }
        };
        let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
        Ok(Spanned::new(
            span,
            Expr::Await(kind, Box::new(Spanned::new(span, body))),
        ))
    }

    fn parse_infer(&mut self, start: LexSpan) -> ParseResult<Spanned<Expr>> {
        self.bump(); // infer
        let ty = self.parse_ty()?;
        self.expect(TokenKind::KwUsing, "`using`")?;
        let model = self.parse_postfix()?;
        self.expect(TokenKind::LBrace, "`{`")?;
        let mut goal = None;
        let mut input = None;
        let mut constraints = Vec::new();
        let mut rubric = None;
        let mut choices = None;
        let mut validate = None;
        let mut accept = None;
        while !self.at(TokenKind::RBrace) {
            match self.peek_kind() {
                Some(TokenKind::KwGoal) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    goal = Some(Box::new(self.parse_expr()?));
                }
                Some(TokenKind::KwInput) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    input = Some(Box::new(self.parse_expr()?));
                }
                Some(TokenKind::KwConstraints) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    self.expect(TokenKind::LBracket, "`[`")?;
                    while !self.at(TokenKind::RBracket) {
                        constraints.push(self.parse_expr()?);
                        if !self.eat(TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(TokenKind::RBracket, "`]`")?;
                }
                Some(TokenKind::KwRubric) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    rubric = Some(Box::new(self.parse_expr()?));
                }
                Some(TokenKind::KwChoices) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    choices = Some(Box::new(self.parse_expr()?));
                }
                Some(TokenKind::KwValidate) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    validate = Some(Box::new(self.parse_expr()?));
                }
                Some(TokenKind::KwAccept) => {
                    self.bump();
                    self.expect(TokenKind::Colon, "`:`")?;
                    accept = Some(Box::new(self.parse_expr()?));
                }
                _ => {
                    let t = self.peek().cloned();
                    if let Some(_t) = t {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
            if !self.eat(TokenKind::Comma) { /* ok */ }
        }
        let end = self.expect(TokenKind::RBrace, "`}`")?.span;
        let spec = InferSpec {
            span: self.span_from(start, end),
            goal,
            input,
            constraints,
            rubric,
            choices,
            validate,
            accept,
            else_: None,
        };
        // optional `accept { ... } else { ... }` (also accept legacy `require`)
        let mut spec = spec;
        if self.eat(TokenKind::KwAccept) || self.eat(TokenKind::KwRequire) {
            self.expect(TokenKind::LBrace, "`{`")?;
            // parse a comma list of expressions; combine via And
            let mut conds = Vec::new();
            while !self.at(TokenKind::RBrace) {
                conds.push(self.parse_expr()?);
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::RBrace, "`}`")?;
            let combined = conds.into_iter().reduce(|a, b| {
                let span = a.span.union(b.span);
                Spanned::new(
                    span,
                    Expr::Bin {
                        op: BinOp::And,
                        lhs: Box::new(a),
                        rhs: Box::new(b),
                    },
                )
            });
            spec.accept = combined.map(Box::new);
        }
        if self.eat(TokenKind::KwElse) {
            self.expect(TokenKind::LBrace, "`{`")?;
            spec.else_ = Some(self.parse_block()?);
            self.expect(TokenKind::RBrace, "`}`")?;
        }
        let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(end));
        Ok(Spanned::new(
            span,
            Expr::Infer {
                ty: Box::new(ty),
                model: Box::new(model),
                spec,
            },
        ))
    }

    fn parse_decide(&mut self, start: LexSpan) -> ParseResult<Spanned<Expr>> {
        self.bump(); // decide
        let ty = self.parse_ty()?;
        self.expect(TokenKind::KwFrom, "`from`")?;
        let source = self.parse_expr()?;
        // `score by [...]` (weighted) or `lex by [...]` (lexicographic)
        let _is_lex =
            if self.at(TokenKind::Ident) && self.peek().map(|t| t.text.as_str()) == Some("lex") {
                self.bump();
                true
            } else {
                self.expect(TokenKind::KwScore, "`score` or `lex`")?;
                false
            };
        self.expect(TokenKind::KwBy, "`by`")?;
        self.expect(TokenKind::LBracket, "`[`")?;
        let mut score_by = Vec::new();
        while !self.at(TokenKind::RBracket) {
            let cstart = self.peek().map(|t| t.span).unwrap_or(LexSpan::dummy());
            // Optional weight: `0.6: field desc` or just `field desc`
            let weight = if (self.at(TokenKind::Decimal) || self.at(TokenKind::Int))
                && self.peek2().map(|t| t.kind) == Some(TokenKind::Colon)
            {
                let w = self.parse_expr()?;
                self.expect(TokenKind::Colon, "`:`")?;
                Some(w)
            } else {
                None
            };
            let field = self.path()?;
            let dir = if self.at(TokenKind::Ident)
                && self.peek().map(|t| t.text.as_str()) == Some("desc")
            {
                self.bump();
                SortDir::Desc
            } else if self.at(TokenKind::Ident)
                && self.peek().map(|t| t.text.as_str()) == Some("asc")
            {
                self.bump();
                SortDir::Asc
            } else {
                SortDir::Asc
            };
            let cend = self.peek().map(|t| t.span).unwrap_or(cstart);
            score_by.push(ScoreClause {
                span: self.span_from(cstart, cend),
                weight,
                field,
                dir,
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBracket, "`]`")?;
        let require = if self.eat(TokenKind::KwAccept) || self.eat(TokenKind::KwRequire) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let else_ = if self.eat(TokenKind::KwElse) {
            if self.at(TokenKind::LBrace) {
                self.bump();
                let b = self.parse_block()?;
                self.expect(TokenKind::RBrace, "`}`")?;
                Some(b)
            } else {
                // else <statement> without braces
                Some(vec![self.parse_stmt()?])
            }
        } else {
            None
        };
        let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
        Ok(Spanned::new(
            span,
            Expr::Decide {
                ty: Box::new(ty),
                source: Box::new(source),
                score_by,
                require,
                else_,
            },
        ))
    }

    fn parse_spawn(&mut self, start: LexSpan) -> ParseResult<Spanned<Expr>> {
        self.bump(); // spawn
        let agent = self.path()?;
        self.expect(TokenKind::LParen, "`(`")?;
        let args = self.parse_call_args()?;
        self.expect(TokenKind::RParen, "`)`")?;
        let caps = if self.eat(TokenKind::KwWith) {
            self.expect(TokenKind::Ident, "`caps`")?; // caps
            self.expect(TokenKind::LBracket, "`[`")?;
            let mut caps = Vec::new();
            while !self.at(TokenKind::RBracket) {
                caps.push(self.parse_expr()?);
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::RBracket, "`]`")?;
            caps
        } else {
            Vec::new()
        };
        let budget = if self.eat(TokenKind::KwBudget) {
            Some(self.parse_budget()?)
        } else {
            None
        };
        let span = self.span_from(start, self.peek().map(|t| t.span).unwrap_or(start));
        Ok(Spanned::new(
            span,
            Expr::Spawn {
                agent,
                args,
                caps,
                budget,
            },
        ))
    }
}

pub fn parse_module(src: &str, file: u32) -> ParseResult<Module> {
    let toks = act_syntax::lexer::lex(src, file).map_err(|e| ParseError {
        span: e.span,
        message: e.message,
    })?;
    let mut p = Parser::new(toks, file);
    p.parse_module()
}
