//! Registries: declared types, functions, tasks, tools, and models.

use std::collections::HashMap;

use act_syntax::ast::{ExternModel, ExternTool, FnDecl, Item, Module, TypeDecl};

/// Maps a simple type name to its declaration, for model-output coercion.
pub struct TypeRegistry {
    decls: HashMap<String, TypeDecl>,
}

impl TypeRegistry {
    pub fn from_module(m: &Module) -> TypeRegistry {
        let mut decls = HashMap::new();
        for item in &m.items {
            if let Item::TypeDecl(t) = item {
                decls.insert(t.name.node.clone(), t.clone());
            }
        }
        TypeRegistry { decls }
    }

    pub fn get(&self, name: &str) -> Option<&TypeDecl> {
        self.decls.get(name)
    }
}

/// Maps a callable name to its declaration. Covers `fn`, `proc`, and `task`.
#[derive(Default)]
pub struct FnRegistry {
    pub fns: HashMap<String, FnDecl>,
}

impl FnRegistry {
    pub fn from_module(m: &Module) -> FnRegistry {
        let mut fns = HashMap::new();
        for item in &m.items {
            match item {
                Item::Fn(d) | Item::Proc(d) | Item::Task(d) => {
                    fns.insert(d.name.node.clone(), d.as_ref().clone());
                }
                _ => {}
            }
        }
        FnRegistry { fns }
    }

    pub fn get(&self, name: &str) -> Option<&FnDecl> {
        self.fns.get(name)
    }
}

/// Maps a tool path prefix (e.g. `gh`) to its extern declaration, and model
/// aliases to model decls. Used by the host to dispatch calls.
#[derive(Default)]
pub struct ExternRegistry {
    pub tools: HashMap<String, ExternTool>,
    pub models: HashMap<String, ExternModel>,
}

impl ExternRegistry {
    pub fn from_module(m: &Module) -> ExternRegistry {
        let mut tools = HashMap::new();
        let mut models = HashMap::new();
        for item in &m.items {
            match item {
                Item::ExternTool(t) => {
                    if let Some(prefix) = t.path.first() {
                        tools.insert(prefix.node.clone(), t.as_ref().clone());
                    }
                }
                Item::ExternModel(m) => {
                    let key = m
                        .alias
                        .as_ref()
                        .map(|a| a.node.clone())
                        .or_else(|| m.path.last().map(|i| i.node.clone()));
                    if let Some(k) = key {
                        models.insert(k, m.as_ref().clone());
                    }
                }
                _ => {}
            }
        }
        ExternRegistry { tools, models }
    }
}
