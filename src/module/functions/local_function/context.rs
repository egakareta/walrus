//! Context needed when validating instructions and constructing our `Instr` IR.

use crate::error::{ErrorKind, Result};
use crate::ir::{BlockKind, Instr, InstrLocId, InstrSeq, InstrSeqId, InstrSeqType};
use crate::module::functions::{FunctionId, LocalFunction};
use crate::module::Module;
use crate::parse::IndicesToIds;
use crate::ty::ValType;
use crate::TypeId;
use anyhow::Context;

#[derive(Debug)]
pub(crate) struct ControlFrame {
    /// The block signature, shared with the instruction sequence. Keeping the
    /// signature avoids allocating parameter/result slices for every frame.
    pub ty: InstrSeqType,

    /// If `true`, then this frame is unreachable. This is used to handle
    /// stack-polymorphic typing after unconditional branches.
    pub unreachable: bool,

    /// The id of this control frame's block.
    pub block: InstrSeqId,

    /// This control frame's kind of block, eg loop vs block vs if/else.
    pub kind: BlockKind,

    /// Where this block's instructions start in `ValidationContext::instrs`.
    pub start: usize,
}

/// The control frame stack.
pub(crate) type ControlStack = Vec<ControlFrame>;

#[derive(Debug)]
pub(crate) struct ValidationContext<'a> {
    /// The module that we're adding a function for.
    pub module: &'a Module,

    /// Mapping of indexes back to ids.
    pub indices: &'a IndicesToIds,

    /// This function's local index space, resolved once for the whole body.
    pub locals: &'a [crate::LocalId],

    /// The function being validated/constructed.
    pub func: &'a mut LocalFunction,

    /// The control frames stack.
    pub controls: &'a mut ControlStack,

    /// The instructions of every open block, innermost last. Each frame's
    /// instructions start at its `start`; they are moved into the frame's
    /// `InstrSeq`, at their exact size, when the frame is popped. This avoids
    /// growing every block's `Vec` one push at a time.
    pub instrs: &'a mut Vec<(Instr, InstrLocId)>,

    /// If we're currently parsing an if/else instruction, where we're at
    pub if_else: Vec<IfElseState>,
}

#[derive(Debug)]
pub struct IfElseState {
    pub start: InstrLocId,
    pub consequent: InstrSeqId,
    pub alternative: Option<InstrSeqId>,
}

impl<'a> ValidationContext<'a> {
    /// Create a new function context.
    pub fn new(
        module: &'a Module,
        indices: &'a IndicesToIds,
        func_id: FunctionId,
        func: &'a mut LocalFunction,
        controls: &'a mut ControlStack,
        instrs: &'a mut Vec<(Instr, InstrLocId)>,
    ) -> ValidationContext<'a> {
        debug_assert!(instrs.is_empty());
        ValidationContext {
            module,
            indices,
            locals: indices.function_locals(func_id),
            func,
            controls,
            instrs,
            if_else: Vec::new(),
        }
    }

    pub fn push_control(&mut self, kind: BlockKind, ty: InstrSeqType) -> InstrSeqId {
        let block = self.func.add_block(|id| InstrSeq::new(id, ty));
        self.controls.push(ControlFrame {
            ty,
            unreachable: false,
            block,
            kind,
            start: self.instrs.len(),
        });
        block
    }

    pub fn push_control_with_ty(&mut self, kind: BlockKind, ty: TypeId) -> InstrSeqId {
        self.push_control(kind, ty.into())
    }

    pub fn catch_type(&self, frame: &ControlFrame, params: &[ValType]) -> Result<InstrSeqType> {
        let results = match &frame.ty {
            InstrSeqType::Simple(None) => &[],
            InstrSeqType::Simple(Some(ty)) => std::slice::from_ref(ty),
            InstrSeqType::MultiValue(ty) => self.module.types.results(*ty),
        };
        InstrSeqType::existing(&self.module.types, params, results)
            .ok_or_else(|| anyhow::anyhow!("catch handler type was not registered"))
    }

    pub fn pop_control(&mut self) -> Result<(ControlFrame, InstrSeqId)> {
        let frame = impl_pop_control(self.controls)?;
        let block = frame.block;
        // The outer frame often owns the complete scratch buffer. Transfer
        // it instead of allocating and copying all instructions just to drop
        // that buffer when parsing this function finishes. Avoid retaining a
        // large buffer for a tiny parent after extracting a large child.
        self.func.block_mut(block).instrs = if frame.start == 0
            && self.instrs.capacity() <= self.instrs.len().saturating_mul(2).max(8)
        {
            std::mem::take(self.instrs)
        } else {
            self.instrs.split_off(frame.start)
        };
        Ok((frame, block))
    }

    pub fn unreachable(&mut self) {
        let frame = self.controls.last_mut().unwrap();
        frame.unreachable = true;
    }

    pub fn control(&self, n: usize) -> Result<&ControlFrame> {
        if n >= self.controls.len() {
            anyhow::bail!("jump to nonexistent control block");
        }
        let idx = self.controls.len() - n - 1;
        Ok(&self.controls[idx])
    }

    pub fn alloc_instr_in_control(
        &mut self,
        control: usize,
        instr: impl Into<Instr>,
        loc: InstrLocId,
    ) -> Result<()> {
        let frame = self.control(control)?;
        if frame.unreachable {
            return Ok(());
        }
        // Frames inside `control` have no instructions yet (this is used to
        // add a block instruction to its parent right after pushing the
        // block's frame), so the parent's instructions end at the top of
        // `instrs`. Push there and move the inner frames' starts past it.
        let inner = self.controls.len() - control..;
        debug_assert!(self.controls[inner.clone()]
            .iter()
            .all(|f| f.start == self.instrs.len()));
        self.instrs.push((instr.into(), loc));
        for f in &mut self.controls[inner] {
            f.start += 1;
        }
        Ok(())
    }

    #[inline]
    pub fn alloc_instr(&mut self, instr: impl Into<Instr>, loc: InstrLocId) {
        // The common case appends to the innermost frame. No enclosing
        // frame's start offset needs adjusting here.
        if !self.controls.last().unwrap().unreachable {
            self.instrs.push((instr.into(), loc));
        }
    }

    pub fn add_legacy_catch(&mut self, catch: crate::ir::LegacyCatch) -> Result<()> {
        // Find the most recent Try instruction in the parent control block
        let frame = self.control(1)?; // Parent block, not the try block itself

        // A `try` in unreachable code was never added to its parent block
        // (`alloc_instr_in_control` drops it), so its clauses have no `Try`
        // to attach to and are dropped along with it.
        if frame.unreachable {
            return Ok(());
        }
        let parent_start = frame.start;

        // The Try instruction should be the last instruction in the parent
        // block, which ends where the innermost frame starts: the try body for
        // `delegate`, or the new (still empty) handler for `catch`.
        let end = self.controls.last().unwrap().start;
        if end > parent_start {
            if let (Instr::Try(ref mut try_instr), _) = self.instrs[end - 1] {
                try_instr.catches.push(catch);
                return Ok(());
            }
        }

        anyhow::bail!("No Try instruction found to add catch clause to");
    }
}

fn impl_pop_control(controls: &mut ControlStack) -> Result<ControlFrame> {
    controls
        .last()
        .ok_or(ErrorKind::InvalidWasm)
        .context("attempted to pop a frame from an empty control stack")?;
    let frame = controls.pop().unwrap();
    Ok(frame)
}
