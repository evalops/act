//! Canonical formatter for Act.
//!
//! Produces canonical source text from an AST. The output is designed to be
//! idempotent: formatting an already-formatted module yields the same text, and
//! `parse(format(parse(src))) == parse(src)` structurally. Parenthesization is
//! minimal but sufficient to preserve the original operator tree.

#![allow(clippy::needless_lifetimes)]

use act_syntax::ast::*;

/// Format a module to canonical source text.
pub fn format_module(m: &Module) -> String {
    let mut f = Formatter { out: String::new() };
    f.module(m);
    f.out
}

struct Formatter {
    out: String,
}

impl Formatter {
    fn s(&mut self, st: &str) {
        self.out.push_str(st);
    }
    fn pad(&mut self, ind: usize) {
        for _ in 0..ind {
            self.out.push_str("  ");
        }
    }

    fn path(&mut self, p: &[Ident]) {
        for (i, seg) in p.iter().enumerate() {
            if i > 0 {
                self.s(".");
            }
            self.s(&seg.node);
        }
    }

    // ---- module / header ----

    fn module(&mut self, m: &Module) {
        // header
        self.s("module ");
        self.path(&m.header.name);
        if let Some(v) = &m.header.version {
            self.s("@");
            self.s(v);
        }
        self.s("\n");
        if !m.header.uses.is_empty() {
            self.s("\n");
            for u in &m.header.uses {
                self.use_decl(u);
                self.s("\n");
            }
        }
        for (i, item) in m.items.iter().enumerate() {
            self.s("\n");
            // blank line already added; if not first and there were uses, we still want separation
            let _ = i;
            self.item(item, 0);
        }
    }

    fn use_decl(&mut self, u: &UseDecl) {
        match u.kind {
            UseKind::Use => self.s("use "),
            UseKind::Tool => self.s("use tool "),
            UseKind::Lib => self.s("use lib "),
            UseKind::Model => self.s("use model "),
        }
        self.path(&u.path);
        if let Some(v) = &u.version {
            self.s("@");
            self.s(v);
        }
        if let Some(a) = &u.alias {
            self.s(" as ");
            self.s(&a.node);
        }
    }

    // ---- items ----

    fn item(&mut self, item: &Item, ind: usize) {
        match item {
            Item::TypeDecl(t) => self.type_decl(t, ind),
            Item::Fn(d) => self.fn_decl(d, ind),
            Item::Proc(d) => self.fn_decl(d, ind),
            Item::Task(d) => self.fn_decl(d, ind),
            Item::Agent(a) => self.agent_decl(a, ind),
            Item::ExternTool(t) => self.extern_tool(t, ind),
            Item::ExternModel(t) => self.extern_model(t, ind),
            Item::Test(t) => self.test_block("test", t, ind),
            Item::Eval(t) => self.test_block("eval", t, ind),
        }
    }

    fn type_decl(&mut self, t: &TypeDecl, ind: usize) {
        self.pad(ind);
        self.s("type ");
        self.s(&t.name.node);
        // body
        match &t.body {
            TypeBody::Record(fields) => {
                self.s(" = ");
                if fields.is_empty() {
                    self.s("{}");
                } else {
                    self.s("{\n");
                    for field in fields {
                        self.pad(ind + 1);
                        self.s(&field.name.node);
                        if field.optional {
                            self.s("?");
                        }
                        self.s(": ");
                        self.ty(&field.ty.node);
                        self.s(",\n");
                    }
                    self.pad(ind);
                    self.s("}");
                }
            }
            TypeBody::Enum(variants) => {
                self.s(" =\n");
                for (i, v) in variants.iter().enumerate() {
                    self.pad(ind + 1);
                    self.s("| ");
                    self.s(&v.name.node);
                    if !v.fields.is_empty() {
                        self.s("(");
                        for (j, (fname, fty)) in v.fields.iter().enumerate() {
                            if j > 0 {
                                self.s(", ");
                            }
                            if let Some(n) = fname {
                                self.s(&n.node);
                                self.s(": ");
                            }
                            self.ty(&fty.node);
                        }
                        self.s(")");
                    }
                    if i + 1 < variants.len() {
                        self.s("\n");
                    }
                }
            }
            TypeBody::Refinement { ty, predicates } => {
                self.s(" = ");
                self.ty(&ty.node);
                if !predicates.is_empty() {
                    self.s(" where ");
                    for (i, p) in predicates.iter().enumerate() {
                        if i > 0 {
                            self.s(" ");
                        }
                        self.expr(&p.node, 0, ind);
                    }
                }
            }
            TypeBody::Alias(ty) => {
                self.s(" = ");
                self.ty(&ty.node);
            }
            TypeBody::Opaque => {}
        }
        self.s("\n");
    }

    fn ty(&mut self, ty: &Ty) {
        match ty {
            Ty::Named { path, args } => {
                self.path(path);
                if !args.is_empty() {
                    self.s("<");
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            self.s(", ");
                        }
                        self.ty(&a.node);
                    }
                    self.s(">");
                }
            }
            Ty::Array(inner) => {
                self.s("[");
                self.ty(&inner.node);
                self.s("]");
            }
            Ty::Map(k, v) => {
                self.s("Map<");
                self.ty(&k.node);
                self.s(", ");
                self.ty(&v.node);
                self.s(">");
            }
            Ty::Set(inner) => {
                self.s("Set<");
                self.ty(&inner.node);
                self.s(">");
            }
            Ty::Tuple(elems) => {
                self.s("(");
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        self.s(", ");
                    }
                    self.ty(&e.node);
                }
                self.s(")");
            }
            Ty::Typeof(e) => {
                self.s("typeof ");
                self.expr(&e.node, 0, 0);
            }
            Ty::Hole => self.s("??"),
        }
    }

    fn fn_decl(&mut self, d: &FnDecl, ind: usize) {
        self.pad(ind);
        match d.kind {
            FnKind::Fn => self.s("fn "),
            FnKind::Proc => self.s("proc "),
            FnKind::Task => self.s("task "),
        }
        self.s(&d.name.node);
        if !d.generics.is_empty() {
            self.s("<");
            for (i, g) in d.generics.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.s(&g.node);
            }
            self.s(">");
        }
        self.s("(");
        for (i, p) in d.params.iter().enumerate() {
            if i > 0 {
                self.s(", ");
            }
            if p.is_cap {
                self.s("cap ");
            }
            self.s(&p.name.node);
            self.s(": ");
            self.ty(&p.ty.node);
            if let Some(def) = &p.default {
                self.s(" = ");
                self.expr(&def.node, 0, ind);
            }
        }
        self.s(") -> ");
        self.ty(&d.return_ty.node);
        self.s("\n");

        // clauses, each indented one level
        if !d.effects.is_empty() {
            self.pad(ind + 1);
            self.s("effects [");
            for (i, e) in d.effects.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.path(&e.node.path);
            }
            self.s("]\n");
        }
        if !d.needs.is_empty() {
            self.pad(ind + 1);
            self.s("needs [\n");
            for cap in &d.needs {
                self.pad(ind + 2);
                self.s("cap ");
                self.path(&cap.node.path);
                if !cap.node.args.is_empty() {
                    self.s("(");
                    for (i, a) in cap.node.args.iter().enumerate() {
                        if i > 0 {
                            self.s(", ");
                        }
                        self.expr(&a.node, 0, ind + 2);
                    }
                    self.s(")");
                }
                self.s(",\n");
            }
            self.pad(ind + 1);
            self.s("]\n");
        }
        if let Some(b) = &d.budget {
            self.budget(b, ind + 1);
        }
        if let Some(p) = &d.policy_expect {
            self.policy_expect(p, ind + 1);
        }

        if let Some(body) = &d.body {
            self.pad(ind);
            self.s("{\n");
            self.block(body, ind + 1);
            self.pad(ind);
            self.s("}\n");
        }
    }

    fn budget(&mut self, b: &Budget, ind: usize) {
        self.pad(ind);
        self.s("budget {\n");
        for lim in &b.limits {
            self.pad(ind + 1);
            self.budget_metric(lim.metric);
            self.s(" ");
            self.budget_op(lim.op);
            self.s(" ");
            self.expr(&lim.value.node, 0, ind + 1);
            self.s(",\n");
        }
        self.pad(ind);
        self.s("}\n");
    }

    fn budget_metric(&mut self, m: BudgetMetric) {
        match m {
            BudgetMetric::WallTime => self.s("wall_time"),
            BudgetMetric::Tokens => self.s("tokens"),
            BudgetMetric::Cost => self.s("cost"),
            BudgetMetric::ToolCalls => self.s("tool_calls"),
        }
    }

    fn budget_op(&mut self, o: BudgetOp) {
        match o {
            BudgetOp::Le => self.s("<="),
            BudgetOp::Lt => self.s("<"),
            BudgetOp::Ge => self.s(">="),
            BudgetOp::Gt => self.s(">"),
            BudgetOp::Eq => self.s("=="),
        }
    }

    fn policy_expect(&mut self, p: &Spanned<PolicyExpect>, ind: usize) {
        self.pad(ind);
        self.s("policy_expect {\n");
        for c in &p.node.clauses {
            self.pad(ind + 1);
            match c.verb {
                PolicyVerb::May => self.s("may "),
                PolicyVerb::MustNot => self.s("must_not "),
                PolicyVerb::RequireHuman => self.s("require_human"),
            }
            if c.verb == PolicyVerb::RequireHuman {
                if let Some(w) = &c.where_clause {
                    self.s(" when ");
                    self.expr(&w.node, 0, ind + 1);
                }
            } else {
                self.path(&c.target);
                if let Some(w) = &c.where_clause {
                    self.s(" where ");
                    self.expr(&w.node, 0, ind + 1);
                }
            }
            self.s("\n");
        }
        self.pad(ind);
        self.s("}\n");
    }

    fn agent_decl(&mut self, a: &AgentDecl, ind: usize) {
        self.pad(ind);
        self.s("agent ");
        self.s(&a.name.node);
        if let Some(st) = &a.state_ty {
            self.s(": ");
            self.ty(&st.node);
        }
        self.s("\n");
        if !a.effects.is_empty() {
            self.pad(ind + 1);
            self.s("effects [");
            for (i, e) in a.effects.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.path(&e.node.path);
            }
            self.s("]\n");
        }
        if !a.needs.is_empty() {
            self.pad(ind + 1);
            self.s("needs [\n");
            for cap in &a.needs {
                self.pad(ind + 2);
                self.s("cap ");
                self.path(&cap.node.path);
                self.s(",\n");
            }
            self.pad(ind + 1);
            self.s("]\n");
        }
        if let Some(b) = &a.budget {
            self.budget(b, ind + 1);
        }
        for h in &a.handlers {
            self.event_handler(h, ind);
        }
    }

    fn event_handler(&mut self, h: &EventHandler, ind: usize) {
        self.pad(ind);
        self.s("on ");
        match h.trigger.kind {
            EventKind::OnMessage => self.s("message "),
            EventKind::On => {}
        }
        self.path(&h.trigger.path);
        self.s(" as ");
        self.s(&h.binder.node);
        if let Some(w) = &h.where_clause {
            self.s(" where ");
            self.expr(&w.node, 0, ind);
        }
        self.s(" {\n");
        self.block(&h.body, ind + 1);
        self.pad(ind);
        self.s("}\n");
    }

    fn extern_tool(&mut self, t: &ExternTool, ind: usize) {
        self.pad(ind);
        self.s("extern tool ");
        self.path(&t.path);
        if !t.params.is_empty() {
            self.s("(");
            for (i, p) in t.params.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.s(&p.name.node);
                self.s(": ");
                self.ty(&p.ty.node);
            }
            self.s(")");
        }
        self.s(" -> ");
        self.ty(&t.return_ty.node);
        if !t.effects.is_empty() {
            self.s(" effects [");
            for (i, e) in t.effects.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.path(&e.node.path);
            }
            self.s("]");
        }
        if let Some(ty) = &t.timeout {
            self.s(" timeout ");
            self.expr(&ty.node, 0, ind);
        }
        if let Some(idem) = &t.idempotent_by {
            self.s(" idempotent by ");
            self.expr(&idem.node, 0, ind);
        }
        if let Some(r) = &t.retry {
            self.s(" retry { attempts: ");
            self.expr(&r.attempts.node, 0, ind);
            self.s(", on: ");
            for (i, p) in r.on.iter().enumerate() {
                if i > 0 {
                    self.s(", ");
                }
                self.path(p);
            }
            self.s(", backoff: ");
            self.expr(&r.backoff.node, 0, ind);
            self.s(" }");
        }
        self.s("\n");
    }

    fn extern_model(&mut self, m: &ExternModel, ind: usize) {
        let _ = ind;
        self.s("extern model ");
        self.path(&m.path);
        if let Some(a) = &m.alias {
            self.s(" as ");
            self.s(&a.node);
        }
        self.s("\n");
    }

    fn test_block(&mut self, kw: &str, t: &TestBlock, ind: usize) {
        self.pad(ind);
        self.s(kw);
        self.s(" ");
        self.ident_or_str(&t.label.node);
        self.s(" {\n");
        self.block(&t.body, ind + 1);
        self.pad(ind);
        self.s("}\n");
    }

    fn ident_or_str(&mut self, name: &str) {
        if name.is_empty()
            || name.chars().next().unwrap().is_ascii_digit()
            || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            self.s("\"");
            self.s(&escape_string(name));
            self.s("\"");
        } else {
            self.s(name);
        }
    }

    // ---- statements ----

    fn block(&mut self, block: &Block, ind: usize) {
        for stmt in block {
            self.stmt(stmt, ind);
        }
    }

    fn stmt(&mut self, s: &Spanned<Stmt>, ind: usize) {
        match &s.node {
            Stmt::Let { name, ty, init, .. } => {
                self.pad(ind);
                self.s("let ");
                self.s(&name.node);
                if let Some(t) = ty {
                    self.s(": ");
                    self.ty(&t.node);
                }
                self.s(" = ");
                self.expr(&init.node, 0, ind);
                self.s("\n");
            }
            Stmt::Var { name, ty, init } => {
                self.pad(ind);
                self.s("var ");
                self.s(&name.node);
                if let Some(t) = ty {
                    self.s(": ");
                    self.ty(&t.node);
                }
                self.s(" = ");
                self.expr(&init.node, 0, ind);
                self.s("\n");
            }
            Stmt::Assign { target, value } => {
                self.pad(ind);
                self.expr(&target.node, 0, ind);
                self.s(" = ");
                self.expr(&value.node, 0, ind);
                self.s("\n");
            }
            Stmt::Expr(e) => {
                self.pad(ind);
                self.expr(&e.node, 0, ind);
                self.s("\n");
            }
            Stmt::Return(e) => {
                self.pad(ind);
                self.s("return");
                if let Some(e) = e {
                    self.s(" ");
                    self.expr(&e.node, 0, ind);
                }
                self.s("\n");
            }
            Stmt::If {
                cond, then, else_, ..
            } => {
                self.pad(ind);
                self.s("if ");
                self.expr(&cond.node, 0, ind);
                self.s(" {\n");
                self.block(then, ind + 1);
                self.pad(ind);
                self.s("}");
                if let Some(e) = else_ {
                    self.s(" else {\n");
                    self.block(e, ind + 1);
                    self.pad(ind);
                    self.s("}");
                }
                self.s("\n");
            }
            Stmt::For {
                item,
                iter,
                limit,
                body,
            } => {
                self.pad(ind);
                self.s("for ");
                self.s(&item.node);
                self.s(" in ");
                self.expr(&iter.node, 0, ind);
                if let Some(l) = limit {
                    self.s(" limit ");
                    self.expr(&l.node, 0, ind);
                }
                self.s(" {\n");
                self.block(body, ind + 1);
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::While { cond, max, body } => {
                self.pad(ind);
                self.s("while ");
                self.expr(&cond.node, 0, ind);
                if let Some(m) = max {
                    self.s(" max ");
                    self.expr(&m.node, 0, ind);
                }
                self.s(" {\n");
                self.block(body, ind + 1);
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::Match { scrutinee, arms } => {
                self.pad(ind);
                self.s("match ");
                self.expr(&scrutinee.node, 0, ind);
                self.s(" {\n");
                for arm in arms {
                    self.pad(ind + 1);
                    self.pattern(&arm.pattern.node);
                    if let Some(g) = &arm.guard {
                        self.s(" where ");
                        self.expr(&g.node, 0, ind + 1);
                    }
                    self.s(" => {\n");
                    self.block(&arm.body, ind + 2);
                    self.pad(ind + 1);
                    self.s("}\n");
                }
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::Recover {
                error_ty,
                from,
                body,
            } => {
                self.pad(ind);
                self.s("recover ");
                self.path(&error_ty.node);
                self.s(" from ");
                self.expr(&from.node, 0, ind);
                self.s(" {\n");
                self.block(body, ind + 1);
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::Defer { kind, body } => {
                self.pad(ind);
                self.s("defer ");
                match kind {
                    DeferKind::Compensate => self.s("compensate"),
                }
                self.s(" {\n");
                self.block(body, ind + 1);
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::Require(e) => {
                self.pad(ind);
                self.s("require ");
                self.expr(&e.node, 0, ind);
                self.s("\n");
            }
            Stmt::Check { cond, else_block } => {
                self.pad(ind);
                self.s("check ");
                self.expr(&cond.node, 0, ind);
                if let Some(e) = else_block {
                    self.s(" else {\n");
                    self.block(e, ind + 1);
                    self.pad(ind);
                    self.s("}");
                }
                self.s("\n");
            }
            Stmt::Ensure(e) => {
                self.pad(ind);
                self.s("ensure ");
                self.expr(&e.node, 0, ind);
                self.s("\n");
            }
            Stmt::Trace { label, fields } => {
                self.pad(ind);
                self.s("trace ");
                self.ident_or_str(&label.node);
                self.s(" {\n");
                for (n, v) in fields {
                    self.pad(ind + 1);
                    self.s(&n.node);
                    self.s(": ");
                    self.expr(&v.node, 0, ind + 1);
                    self.s(",\n");
                }
                self.pad(ind);
                self.s("}\n");
            }
            Stmt::Checkpoint {
                label,
                body,
                require,
            } => {
                self.pad(ind);
                self.s("checkpoint ");
                self.s(&label.node);
                self.s(" ");
                self.expr(&body.node, 0, ind);
                if let Some(r) = require {
                    self.s(" require ");
                    self.expr(&r.node, 0, ind);
                }
                self.s("\n");
            }
            Stmt::Invariant {
                label,
                before,
                require,
            } => {
                self.pad(ind);
                self.s("invariant ");
                self.s(&label.node);
                if let Some(b) = before {
                    self.s(" before ");
                    self.path(b);
                }
                self.s(" require ");
                self.expr(&require.node, 0, ind);
                self.s("\n");
            }
        }
    }

    fn pattern(&mut self, p: &Pattern) {
        match p {
            Pattern::Tag { name, binder } => {
                self.path(name);
                if let Some(b) = binder {
                    self.s("(");
                    self.s(&b.node);
                    self.s(")");
                }
            }
            Pattern::Bind(i) => self.s(&i.node),
            Pattern::Wildcard => self.s("_"),
            Pattern::Lit(e) => self.expr(&e.node, 0, 0),
        }
    }

    // ---- expressions ----

    fn expr(&mut self, e: &Expr, min_bp: u8, ind: usize) {
        if let Expr::Bin { op, lhs, rhs } = e {
            let bp = binop_bp(*op);
            if bp < min_bp {
                self.s("(");
                self.expr(&lhs.node, bp, ind);
                self.s(" ");
                self.binop_str(op);
                self.s(" ");
                self.expr(&rhs.node, bp + 1, ind);
                self.s(")");
                return;
            }
            self.expr(&lhs.node, bp, ind);
            self.s(" ");
            self.binop_str(op);
            self.s(" ");
            self.expr(&rhs.node, bp + 1, ind);
            return;
        }
        if let Expr::Un { op, expr } = e {
            match op {
                UnOp::Not => self.s("!"),
                UnOp::Neg => self.s("-"),
            }
            self.expr(&expr.node, 6, ind);
            return;
        }
        self.atom(e, ind);
    }

    fn binop_str(&mut self, op: &BinOp) {
        match op {
            BinOp::Add => self.s("+"),
            BinOp::Sub => self.s("-"),
            BinOp::Mul => self.s("*"),
            BinOp::Div => self.s("/"),
            BinOp::Mod => self.s("%"),
            BinOp::Eq => self.s("=="),
            BinOp::Ne => self.s("!="),
            BinOp::Lt => self.s("<"),
            BinOp::Le => self.s("<="),
            BinOp::Gt => self.s(">"),
            BinOp::Ge => self.s(">="),
            BinOp::And => self.s("&&"),
            BinOp::Or => self.s("||"),
            BinOp::In => self.s("in"),
            BinOp::NotIn => self.s("not in"),
            BinOp::Pipe => self.s("|>"),
        }
    }

    fn atom(&mut self, e: &Expr, ind: usize) {
        match e {
            Expr::Lit(l) => self.literal(l),
            Expr::Path(p) => self.path(p),
            Expr::Interp(parts) | Expr::Markdown(parts) => {
                self.s("\"");
                for part in parts {
                    match part {
                        InterpPart::Str(s) => self.s(&escape_string(s)),
                        InterpPart::Expr(e) => {
                            self.s("${");
                            self.expr(&e.node, 0, ind);
                            self.s("}");
                        }
                    }
                }
                self.s("\"");
            }
            Expr::Call { callee, args } => {
                self.expr(&callee.node, 7, ind);
                self.call_args(args, ind);
            }
            Expr::Method {
                receiver,
                name,
                args,
            } => {
                self.expr(&receiver.node, 7, ind);
                self.s(".");
                self.s(&name.node);
                self.call_args(args, ind);
            }
            Expr::Field { receiver, name } => {
                self.expr(&receiver.node, 7, ind);
                self.s(".");
                self.s(&name.node);
            }
            Expr::Index { receiver, index } => {
                self.expr(&receiver.node, 7, ind);
                self.s("[");
                self.expr(&index.node, 0, ind);
                self.s("]");
            }
            Expr::Try(e) => {
                self.s("try ");
                self.expr(&e.node, 6, ind);
            }
            Expr::ResultCtor { variant, value } => {
                match variant {
                    ResultVariant::Ok => self.s("ok"),
                    ResultVariant::Err => self.s("err"),
                }
                if let Some(v) = value {
                    self.s("(");
                    self.expr(&v.node, 0, ind);
                    self.s(")");
                }
            }
            Expr::Record(fields) => self.record(fields, ind),
            Expr::Array(elems) => self.array(elems, ind),
            Expr::Block(b) => self.block_expr(b, ind),
            Expr::ParallelRecord(fields) => {
                self.s("all ");
                self.record(fields, ind);
            }
            Expr::Await(kind, body) => self.await_expr(*kind, &body.node, ind),
            Expr::Infer { ty, model, spec } => self.infer(ty, model, spec, ind),
            Expr::Decide {
                ty,
                source,
                score_by,
                require,
                else_,
            } => self.decide(
                ty,
                source,
                score_by,
                require.as_deref(),
                else_.as_ref(),
                ind,
            ),
            Expr::Spawn {
                agent,
                args,
                caps,
                budget,
            } => self.spawn(agent, args, caps, budget.as_ref(), ind),
            Expr::Hole(h) => self.hole(h, ind),
            Expr::Replay { label } => {
                self.s("replay trace(");
                self.expr(&label.node, 0, ind);
                self.s(")");
            }
            // Bin/Un are handled by expr(); keep arms for exhaustiveness.
            Expr::Bin { .. } | Expr::Un { .. } => unreachable!("handled by expr()"),
        }
    }

    fn call_args(&mut self, args: &[CallArg], ind: usize) {
        self.s("(");
        for (i, a) in args.iter().enumerate() {
            if i > 0 {
                self.s(", ");
            }
            if let Some(n) = &a.name {
                self.s(&n.node);
                self.s(": ");
            }
            self.expr(&a.value.node, 0, ind);
        }
        self.s(")");
    }

    fn record(&mut self, fields: &[(Ident, Spanned<Expr>)], ind: usize) {
        if fields.is_empty() {
            self.s("{}");
            return;
        }
        self.s("{\n");
        for (n, v) in fields {
            self.pad(ind + 1);
            self.s(&n.node);
            self.s(": ");
            self.expr(&v.node, 0, ind + 1);
            self.s(",\n");
        }
        self.pad(ind);
        self.s("}");
    }

    fn array(&mut self, elems: &[Spanned<Expr>], ind: usize) {
        if elems.is_empty() {
            self.s("[]");
            return;
        }
        self.s("[\n");
        for e in elems {
            self.pad(ind + 1);
            self.expr(&e.node, 0, ind + 1);
            self.s(",\n");
        }
        self.pad(ind);
        self.s("]");
    }

    fn block_expr(&mut self, b: &Block, ind: usize) {
        self.s("{\n");
        self.block(b, ind + 1);
        self.pad(ind);
        self.s("}");
    }

    fn await_expr(&mut self, kind: AwaitKind, body: &AwaitBody, ind: usize) {
        match kind {
            AwaitKind::All => {
                self.s("await all ");
                if let AwaitBody::All(branches) = body {
                    self.record(branches, ind);
                }
            }
            AwaitKind::Map => {
                if let AwaitBody::Map {
                    item,
                    iter,
                    parallel,
                    limit,
                    body,
                } = body
                {
                    self.s("await map ");
                    self.s(&item.node);
                    self.s(" in ");
                    self.expr(&iter.node, 0, ind);
                    if let Some(p) = parallel {
                        self.s(" parallel ");
                        self.expr(&p.node, 0, ind);
                    }
                    if let Some(l) = limit {
                        self.s(" limit ");
                        self.expr(&l.node, 0, ind);
                    }
                    self.s(" {\n");
                    self.block(body, ind + 1);
                    self.pad(ind);
                    self.s("}");
                }
            }
            AwaitKind::Race => {
                if let AwaitBody::Race { branches, timeout } = body {
                    self.s("await race first_ok ");
                    self.record(branches, ind);
                    if let Some(t) = timeout {
                        self.s(" timeout ");
                        self.expr(&t.node, 0, ind);
                    }
                }
            }
            AwaitKind::Quorum => {
                if let AwaitBody::Quorum {
                    quorum,
                    of,
                    branches,
                } = body
                {
                    self.s("await quorum ");
                    self.expr(&quorum.node, 0, ind);
                    self.s(" of ");
                    self.expr(&of.node, 0, ind);
                    self.s(" ");
                    self.record(branches, ind);
                }
            }
        }
    }

    fn infer(&mut self, ty: &Spanned<Ty>, model: &Spanned<Expr>, spec: &InferSpec, ind: usize) {
        self.s("infer ");
        self.ty(&ty.node);
        self.s(" using ");
        self.expr(&model.node, 7, ind);
        self.s(" {\n");
        if let Some(g) = &spec.goal {
            self.field_expr("goal", &g.node, ind + 1);
        }
        if let Some(i) = &spec.input {
            self.field_expr("input", &i.node, ind + 1);
        }
        if !spec.constraints.is_empty() {
            self.pad(ind + 1);
            self.s("constraints: [\n");
            for c in &spec.constraints {
                self.pad(ind + 2);
                self.expr(&c.node, 0, ind + 2);
                self.s(",\n");
            }
            self.pad(ind + 1);
            self.s("]\n");
        }
        if let Some(r) = &spec.rubric {
            self.field_expr("rubric", &r.node, ind + 1);
        }
        if let Some(c) = &spec.choices {
            self.field_expr("choices", &c.node, ind + 1);
        }
        if let Some(v) = &spec.validate {
            self.field_expr("validate", &v.node, ind + 1);
        }
        self.pad(ind);
        self.s("}");
        // `accept` is conflated in the AST between the inline `accept:` field and
        // the post-block `accept { ... }` form. Canonicalize to the block form.
        if let Some(a) = &spec.accept {
            self.s(" accept {\n");
            self.pad(ind + 1);
            self.expr(&a.node, 0, ind + 1);
            self.s(",\n");
            self.pad(ind);
            self.s("}");
        }
        if let Some(els) = &spec.else_ {
            self.s(" else {\n");
            self.block(els, ind + 1);
            self.pad(ind);
            self.s("}");
        }
    }

    fn field_expr(&mut self, name: &str, val: &Expr, ind: usize) {
        self.pad(ind);
        self.s(name);
        self.s(": ");
        self.expr(val, 0, ind);
        self.s("\n");
    }

    fn decide(
        &mut self,
        ty: &Spanned<Ty>,
        source: &Spanned<Expr>,
        score_by: &[ScoreClause],
        require: Option<&Spanned<Expr>>,
        else_: Option<&Block>,
        ind: usize,
    ) {
        self.s("decide ");
        self.ty(&ty.node);
        self.s(" from ");
        self.expr(&source.node, 0, ind);
        self.s("\n");
        self.pad(ind + 1);
        self.s("score by [\n");
        for c in score_by {
            self.pad(ind + 2);
            if let Some(w) = &c.weight {
                self.expr(&w.node, 0, ind + 2);
                self.s(": ");
            }
            self.path(&c.field);
            match c.dir {
                SortDir::Asc => self.s(" asc"),
                SortDir::Desc => self.s(" desc"),
            }
            self.s(",\n");
        }
        self.pad(ind + 1);
        self.s("]\n");
        if let Some(r) = require {
            self.pad(ind + 1);
            self.s("accept ");
            self.expr(&r.node, 0, ind + 1);
            self.s("\n");
        }
        if let Some(els) = else_ {
            self.pad(ind + 1);
            self.s("else {\n");
            self.block(els, ind + 2);
            self.pad(ind + 1);
            self.s("}\n");
        }
    }

    fn spawn(
        &mut self,
        agent: &Path,
        args: &[CallArg],
        caps: &[Spanned<Expr>],
        budget: Option<&Budget>,
        ind: usize,
    ) {
        self.s("spawn ");
        self.path(agent);
        self.call_args(args, ind);
        if !caps.is_empty() {
            self.s(" with caps [\n");
            for c in caps {
                self.pad(ind + 1);
                self.expr(&c.node, 0, ind + 1);
                self.s(",\n");
            }
            self.pad(ind);
            self.s("]");
        }
        if let Some(b) = budget {
            self.s("\n");
            self.budget(b, ind);
        }
    }

    fn hole(&mut self, h: &HoleSpec, ind: usize) {
        match h {
            HoleSpec::Plain(hint) => {
                self.s("?? ");
                self.expr(&hint.node, 0, ind);
            }
            HoleSpec::Constrained { goal, must_satisfy } => {
                self.s("?? {\n");
                if let Some(g) = goal {
                    self.field_expr("goal", &g.node, ind + 1);
                }
                if !must_satisfy.is_empty() {
                    self.pad(ind + 1);
                    self.s("must_satisfy: [\n");
                    for m in must_satisfy {
                        self.pad(ind + 2);
                        self.expr(&m.node, 0, ind + 2);
                        self.s(",\n");
                    }
                    self.pad(ind + 1);
                    self.s("]\n");
                }
                self.pad(ind);
                self.s("}");
            }
        }
    }

    fn literal(&mut self, l: &Literal) {
        match l {
            Literal::Int(n) => self.s(&n.to_string()),
            Literal::Decimal(d) => self.s(d),
            Literal::String(s) => {
                self.s("\"");
                self.s(&escape_string(s));
                self.s("\"");
            }
            Literal::Bool(b) => self.s(if *b { "true" } else { "false" }),
            Literal::Duration(d) => self.s(d),
            Literal::Money(a, c) => {
                self.s(a);
                self.s(" ");
                self.s(c);
            }
            Literal::Null => self.s("null"),
        }
    }
}

fn binop_bp(op: BinOp) -> u8 {
    match op {
        BinOp::Or => 1,
        BinOp::And => 2,
        BinOp::Eq
        | BinOp::Ne
        | BinOp::Lt
        | BinOp::Le
        | BinOp::Gt
        | BinOp::Ge
        | BinOp::In
        | BinOp::NotIn => 3,
        BinOp::Add | BinOp::Sub => 4,
        BinOp::Mul | BinOp::Div | BinOp::Mod => 5,
        BinOp::Pipe => 0,
    }
}

fn escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}
