//! Canonical AST for Act.
//!
//! Node IDs are stable and preserved across formatting so that diagnostics,
//! traces, and repair patches can refer to nodes by identity.

#![allow(clippy::needless_lifetimes)]

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

/// Globally unique, stable node id assigned at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct NodeId(pub u64);

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl NodeId {
    pub fn fresh() -> NodeId {
        NodeId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A source span: byte offset range plus file id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
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
    pub fn union(self, other: Span) -> Span {
        Span {
            file: self.file,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// Attaches a span + node id to a payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T> {
    pub id: NodeId,
    pub span: Span,
    pub node: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, node: T) -> Spanned<T> {
        Spanned {
            id: NodeId::fresh(),
            span,
            node,
        }
    }
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            id: self.id,
            span: self.span,
            node: f(self.node),
        }
    }
}

pub type Ident = Spanned<String>;
pub type Path = Vec<Ident>;

/// Top-level module.
#[derive(Clone, Debug)]
pub struct Module {
    pub span: Span,
    pub header: ModuleHeader,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug)]
pub struct ModuleHeader {
    pub span: Span,
    pub name: Path,              // e.g. evalops.fix_regression
    pub version: Option<String>, // @0.1
    pub uses: Vec<UseDecl>,
}

#[derive(Clone, Debug)]
pub struct UseDecl {
    pub span: Span,
    pub kind: UseKind,
    pub path: Path,
    pub version: Option<String>,
    pub alias: Option<Ident>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UseKind {
    Use,
    Tool,
    Lib,
    Model,
}

/// A top-level declaration.
#[derive(Clone, Debug)]
pub enum Item {
    TypeDecl(TypeDecl),
    Fn(FnDecl),
    Proc(FnDecl),
    Task(FnDecl),
    Agent(AgentDecl),
    ExternTool(ExternTool),
    ExternModel(ExternModel),
    Test(TestBlock),
    Eval(TestBlock),
}

#[derive(Clone, Debug)]
pub struct TypeDecl {
    pub span: Span,
    pub name: Ident,
    pub kind: TypeDeclKind,
    pub body: TypeBody,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TypeDeclKind {
    Type,
    OpaqueType,
}

#[derive(Clone, Debug)]
pub enum TypeBody {
    /// `{ field: T, ... }` with optional fields (`name?`)
    Record(Vec<Field>),
    /// `| ok(value: T) | err(error: E)` variants
    Enum(Vec<Variant>),
    /// `where <predicates>`
    Refinement {
        ty: Box<Spanned<Ty>>,
        predicates: Vec<Spanned<Expr>>,
    },
    /// alias `= T`
    Alias(Box<Spanned<Ty>>),
    /// `T` opaque, no body
    Opaque,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub span: Span,
    pub name: Ident,
    pub optional: bool,
    pub ty: Spanned<Ty>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub span: Span,
    pub name: Ident,
    pub payload: Option<Spanned<Ty>>,
}

/// Function declaration. Shared by `fn`, `proc`, `task`.
#[derive(Clone, Debug)]
pub struct FnDecl {
    pub span: Span,
    pub kind: FnKind,
    pub name: Ident,
    pub generics: Vec<Ident>,
    pub params: Vec<Param>,
    pub return_ty: Spanned<Ty>,
    pub effects: Vec<Spanned<EffectRef>>,
    pub needs: Vec<Spanned<Capability>>,
    pub budget: Option<Budget>,
    pub policy_expect: Option<Spanned<PolicyExpect>>,
    pub body: Option<Block>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FnKind {
    Fn,
    Proc,
    Task,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub span: Span,
    pub name: Ident,
    pub is_cap: bool, // `cap pr_cap: gh.PullRequestCreate`
    pub ty: Spanned<Ty>,
    pub default: Option<Spanned<Expr>>,
}

#[derive(Clone, Debug)]
pub struct AgentDecl {
    pub span: Span,
    pub name: Ident,
    pub state_ty: Option<Spanned<Ty>>,
    pub effects: Vec<Spanned<EffectRef>>,
    pub needs: Vec<Spanned<Capability>>,
    pub budget: Option<Budget>,
    pub handlers: Vec<EventHandler>,
}

#[derive(Clone, Debug)]
pub struct EventHandler {
    pub span: Span,
    pub trigger: EventTrigger,
    pub binder: Ident,
    pub where_clause: Option<Spanned<Expr>>,
    pub budget: Option<Budget>,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub struct EventTrigger {
    pub span: Span,
    pub kind: EventKind,
    pub path: Path, // e.g. gh.pull_request.opened
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EventKind {
    On,
    OnMessage,
}

#[derive(Clone, Debug)]
pub struct ExternTool {
    pub span: Span,
    pub path: Path, // gh.create_pull_request
    pub params: Vec<Param>,
    pub return_ty: Spanned<Ty>,
    pub effects: Vec<Spanned<EffectRef>>,
    pub needs: Vec<Spanned<Capability>>,
    pub timeout: Option<Spanned<Expr>>,
    pub idempotent_by: Option<Spanned<Expr>>,
    pub retry: Option<RetrySpec>,
}

#[derive(Clone, Debug)]
pub struct RetrySpec {
    pub span: Span,
    pub attempts: Spanned<Expr>,
    pub on: Vec<Path>,
    pub backoff: Spanned<Expr>,
}

#[derive(Clone, Debug)]
pub struct ExternModel {
    pub span: Span,
    pub path: Path, // codegen@2026-06 -> resolved path
    pub alias: Option<Ident>,
}

#[derive(Clone, Debug)]
pub struct TestBlock {
    pub span: Span,
    pub label: Ident,
    pub body: Block,
}

/// Budget declaration.
#[derive(Clone, Debug)]
pub struct Budget {
    pub span: Span,
    pub limits: Vec<BudgetLimit>,
}

#[derive(Clone, Debug)]
pub struct BudgetLimit {
    pub span: Span,
    pub metric: BudgetMetric,
    pub op: BudgetOp,
    pub value: Spanned<Expr>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BudgetMetric {
    WallTime,
    Tokens,
    Cost,
    ToolCalls,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BudgetOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
}

/// Policy expectations.
#[derive(Clone, Debug)]
pub struct PolicyExpect {
    pub span: Span,
    pub clauses: Vec<PolicyClause>,
}

#[derive(Clone, Debug)]
pub struct PolicyClause {
    pub span: Span,
    pub verb: PolicyVerb,
    pub target: Path,
    pub where_clause: Option<Spanned<Expr>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PolicyVerb {
    May,
    MustNot,
    RequireHuman,
}

/// Types.
#[derive(Clone, Debug)]
pub enum Ty {
    /// Named: Bool, Int, String, Repo, Hypothesis, Secret<String>, ...
    Named { path: Path, args: Vec<Spanned<Ty>> },
    /// Array<T>
    Array(Box<Spanned<Ty>>),
    /// Map<K, V>
    Map(Box<Spanned<Ty>>, Box<Spanned<Ty>>),
    /// Set<T>
    Set(Box<Spanned<Ty>>),
    /// Tuple (A, B, C)
    Tuple(Vec<Spanned<Ty>>),
    /// `typeof expr`
    Typeof(Box<Spanned<Expr>>),
    /// Inference hole `??`
    Hole,
}

/// Effect reference: `gh.read`, `model`, `state`, `network.read`
#[derive(Clone, Debug, PartialEq)]
pub struct EffectRef {
    pub path: Path,
}

/// Capability: `cap gh.repo.read("evalops/orient-search")`
#[derive(Clone, Debug)]
pub struct Capability {
    pub path: Path,
    pub args: Vec<Spanned<Expr>>,
}

/// Statements and expressions share a block.
pub type Block = Vec<Spanned<Stmt>>;

#[derive(Clone, Debug)]
pub enum Stmt {
    Let {
        mutable: bool,
        name: Ident,
        ty: Option<Spanned<Ty>>,
        init: Spanned<Expr>,
    },
    Var {
        name: Ident,
        ty: Option<Spanned<Ty>>,
        init: Spanned<Expr>,
    },
    Assign {
        target: Spanned<Expr>,
        value: Spanned<Expr>,
    },
    Expr(Spanned<Expr>),
    Return(Option<Spanned<Expr>>),
    If {
        cond: Spanned<Expr>,
        then: Block,
        else_: Option<Block>,
    },
    For {
        item: Ident,
        iter: Spanned<Expr>,
        limit: Option<Spanned<Expr>>,
        body: Block,
    },
    While {
        cond: Spanned<Expr>,
        max: Option<Spanned<Expr>>,
        body: Block,
    },
    Match {
        scrutinee: Spanned<Expr>,
        arms: Vec<MatchArm>,
    },
    Recover {
        error_ty: Spanned<Path>,
        from: Spanned<Expr>,
        body: Block,
    },
    Defer {
        kind: DeferKind,
        body: Block,
    },
    Require(Spanned<Expr>),
    Check(Spanned<Expr>),
    Ensure(Spanned<Expr>),
    Trace {
        label: Ident,
        fields: Vec<(Ident, Spanned<Expr>)>,
    },
    Checkpoint {
        label: Ident,
        body: Spanned<Expr>,
        require: Option<Spanned<Expr>>,
    },
    Invariant {
        label: Ident,
        before: Option<Path>,
        require: Spanned<Expr>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DeferKind {
    Compensate,
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub span: Span,
    pub pattern: Spanned<Pattern>,
    pub guard: Option<Spanned<Expr>>,
    pub body: Block,
}

#[derive(Clone, Debug)]
pub enum Pattern {
    /// Variant or enum tag: `ok(x)`, `err(e)`, `some(v)`, `none`
    Tag { name: Path, binder: Option<Ident> },
    /// Literal binding: `x`
    Bind(Ident),
    /// Wildcard `_`
    Wildcard,
    /// Literal value
    Lit(Spanned<Expr>),
}

#[derive(Clone, Debug)]
pub enum Expr {
    /// Literal
    Lit(Literal),
    /// Identifier / path: `input`, `eo.fetch_logs`, `gh.PullRequest`
    Path(Path),
    /// Interpolation string: `${name} latest ...`
    Interp(Vec<InterpPart>),
    /// Call: `foo(a: 1, b: 2)` — mandatory named args for tool/model calls
    Call {
        callee: Box<Spanned<Expr>>,
        args: Vec<CallArg>,
    },
    /// Method-style: `x.summary()`
    Method {
        receiver: Box<Spanned<Expr>>,
        name: Ident,
        args: Vec<CallArg>,
    },
    /// Field access: `input.repo`
    Field {
        receiver: Box<Spanned<Expr>>,
        name: Ident,
    },
    /// Index: `xs[0]`
    Index {
        receiver: Box<Spanned<Expr>>,
        index: Box<Spanned<Expr>>,
    },
    /// Binary op
    Bin {
        op: BinOp,
        lhs: Box<Spanned<Expr>>,
        rhs: Box<Spanned<Expr>>,
    },
    /// Unary op
    Un { op: UnOp, expr: Box<Spanned<Expr>> },
    /// `try expr` (sugar for match)
    Try(Box<Spanned<Expr>>),
    /// `await all { ... }`, `await map x in xs parallel 4 limit 5 { ... }`,
    /// `await race first_ok { ... } timeout 20s`, `await quorum 2 of 3 { ... }`
    Await(AwaitKind, Box<Spanned<AwaitBody>>),
    /// `infer T using model { ... } require { ... } else { ... }`
    Infer {
        ty: Box<Spanned<Ty>>,
        model: Box<Spanned<Expr>>,
        spec: InferSpec,
    },
    /// `decide T from xs score by [...] require ... else ...`
    Decide {
        ty: Box<Spanned<Ty>>,
        source: Box<Spanned<Expr>>,
        score_by: Vec<ScoreClause>,
        require: Option<Box<Spanned<Expr>>>,
        else_: Option<Block>,
    },
    /// `ok(x)`, `err(e)`
    ResultCtor {
        variant: ResultVariant,
        value: Option<Box<Spanned<Expr>>>,
    },
    /// `spawn Type(...) with caps [...] budget { ... }`
    Spawn {
        agent: Path,
        args: Vec<CallArg>,
        caps: Vec<Spanned<Expr>>,
        budget: Option<Budget>,
    },
    /// `?? "hint"` or `?? { goal: ..., must_satisfy: [...] }`
    Hole(HoleSpec),
    /// Record literal: `{ a: 1, b: 2 }`
    Record(Vec<(Ident, Spanned<Expr>)>),
    /// Array literal: `[a, b, c]`
    Array(Vec<Spanned<Expr>>),
    /// Markdown literal: `md""" ... """`
    Markdown(Vec<InterpPart>),
    /// Block expression
    Block(Block),
    /// `all { ... }` desugars to await-less parallel record; kept for await target
    ParallelRecord(Vec<(Ident, Spanned<Expr>)>),
}

#[derive(Clone, Debug)]
pub struct InferSpec {
    pub span: Span,
    pub goal: Option<Box<Spanned<Expr>>>,
    pub input: Option<Box<Spanned<Expr>>>,
    pub constraints: Vec<Spanned<Expr>>,
    pub rubric: Option<Box<Spanned<Expr>>>,
    pub choices: Option<Box<Spanned<Expr>>>,
    pub validate: Option<Box<Spanned<Expr>>>,
    pub accept: Option<Box<Spanned<Expr>>>,
    pub else_: Option<Block>,
}

#[derive(Clone, Debug)]
pub struct ScoreClause {
    pub span: Span,
    pub field: Path,
    pub dir: SortDir,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SortDir {
    Asc,
    Desc,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ResultVariant {
    Ok,
    Err,
}

#[derive(Clone, Debug)]
pub enum HoleSpec {
    Plain(Box<Spanned<Expr>>),
    Constrained {
        goal: Option<Box<Spanned<Expr>>>,
        must_satisfy: Vec<Spanned<Expr>>,
    },
}

#[derive(Clone, Debug)]
pub enum InterpPart {
    Str(String),
    Expr(Spanned<Expr>),
}

#[derive(Clone, Debug)]
pub struct CallArg {
    pub span: Span,
    pub name: Option<Ident>,
    pub value: Spanned<Expr>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    In,
    NotIn,
    Pipe, // |>
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum UnOp {
    Not,
    Neg,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum AwaitKind {
    All,
    Map,
    Race,
    Quorum,
}

#[derive(Clone, Debug)]
pub enum AwaitBody {
    /// `{ logs: <expr>, diff: <expr> }`
    All(Vec<(Ident, Spanned<Expr>)>),
    /// `x in xs parallel 4 limit 5 { body }`
    Map {
        item: Ident,
        iter: Spanned<Expr>,
        parallel: Option<Spanned<Expr>>,
        limit: Option<Spanned<Expr>>,
        body: Block,
    },
    /// `first_ok { a: <expr>, b: <expr> }`
    Race {
        branches: Vec<(Ident, Spanned<Expr>)>,
        timeout: Option<Spanned<Expr>>,
    },
    /// `2 of 3 { a: <expr>, b: <expr>, c: <expr> }`
    Quorum {
        quorum: Spanned<Expr>,
        of: Spanned<Expr>,
        branches: Vec<(Ident, Spanned<Expr>)>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Int(i64),
    Decimal(String),
    String(String),
    Bool(bool),
    /// `30m`, `5s`, `1ms` duration
    Duration(String),
    /// `8.00 USD` money literal -> (amount, currency)
    Money(String, String),
    /// `0.95` score kept as decimal; `true`/`false` handled by Bool
    Null,
}
