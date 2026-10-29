//! Validated numeric function bodies that do not need instruction rewrites.

use super::{EmitFacts, FunctionId, LocalFunction};
use crate::emit::IdsToIndices;
use crate::ir::Local;
use crate::map::{IdHashMap, IdHashSet};
use crate::parse::IndicesToIds;
use crate::{Data, Element, Function, Global, Memory, Module, Table, Type, TypeId, ValType};
use crate::{LocalId, Result};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;
use wasm_encoder::reencode::{self, Reencode};
use wasm_encoder::Encode;
use wasmparser::{
    BlockType, FuncToValidate, FuncValidator, FuncValidatorAllocations, FunctionBody, Operator,
    ValidatorResources, VisitOperator, VisitSimdOperator,
};

#[derive(Debug, Default)]
pub(crate) struct References {
    pub funcs: IdHashSet<Function>,
    pub globals: IdHashSet<Global>,
    pub memories: IdHashSet<Memory>,
    pub tables: IdHashSet<Table>,
    pub types: IdHashSet<Type>,
    pub data: IdHashSet<Data>,
    pub elements: IdHashSet<Element>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BlockEncoding {
    Empty,
    Result(ValType),
    Multi(TypeId),
}

pub(crate) fn block_encodings(module: &Module, indices: &IndicesToIds) -> Vec<BlockEncoding> {
    indices
        .input_types()
        .iter()
        .map(|&id| match module.types.get(id).as_function() {
            Some(f) if f.params().is_empty() && f.results().is_empty() => BlockEncoding::Empty,
            Some(f) if f.params().is_empty() && f.results().len() == 1 => {
                BlockEncoding::Result(f.results()[0])
            }
            _ => BlockEncoding::Multi(id),
        })
        .collect()
}

/// A validated definition kept in its binary form. This variant is created
/// only by the explicit encoded-body configuration, when original instruction
/// metadata is not needed. Resource references and emission facts are computed
/// afresh while parsing. Use `Module::materialize_function` before inspecting or
/// rewriting its individual instructions, before pruning original types.
pub struct EncodedFunction {
    pub(crate) ty: TypeId,
    pub(crate) args: Vec<LocalId>,
    pub(crate) refs: References,
    pub(crate) facts: EmitFacts,
    body: Box<[u8]>,
    offset: usize,
    id: FunctionId,
    indices: Arc<IndicesToIds>,
    blocks: Arc<Vec<BlockEncoding>>,
    validator: FuncToValidate<ValidatorResources>,
}

impl std::fmt::Debug for EncodedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodedFunction")
            .field("ty", &self.ty)
            .field("body_bytes", &self.body.len())
            .field("facts", &self.facts)
            .finish()
    }
}

/// Reusable allocation storage; every function resets all contents.
#[derive(Default)]
pub(super) struct EmitScratch {
    local_indices: Vec<u32>,
    numeric_groups: [Vec<usize>; 5],
    reference_groups: BTreeMap<crate::RefType, Vec<usize>>,
    control: Control,
}

impl EncodedFunction {
    /// The function's signature.
    pub fn ty(&self) -> TypeId {
        self.ty
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ty: TypeId,
        args: Vec<LocalId>,
        scan: Scan,
        body: FunctionBody<'_>,
        id: FunctionId,
        indices: Arc<IndicesToIds>,
        blocks: Arc<Vec<BlockEncoding>>,
        validator: FuncToValidate<ValidatorResources>,
    ) -> Self {
        let offset = body.range().start;
        Self {
            ty,
            args,
            refs: scan.refs,
            facts: scan.facts,
            body: body.as_bytes().into(),
            offset,
            id,
            indices,
            blocks,
            validator,
        }
    }

    pub(crate) fn materialize(&self, module: &Module) -> Result<LocalFunction> {
        let present: IdHashSet<_> = module.types.iter().map(|t| t.id()).collect();
        anyhow::ensure!(
            self.indices
                .input_types()
                .iter()
                .all(|id| present.contains(id)),
            "materialize encoded functions before pruning original types"
        );
        let source = &self.validator;
        let mut validator = FuncToValidate {
            resources: source.resources.clone(),
            index: source.index,
            ty: source.ty,
            features: source.features,
        }
        .into_validator(Default::default());
        let body = FunctionBody::new(wasmparser::BinaryReader::new(&self.body, self.offset));
        let mut locals = body.get_locals_reader()?;
        for _ in 0..locals.get_count() {
            let pos = locals.original_position();
            let (count, ty) = locals.read()?;
            validator.define_locals(pos, count, ty)?;
        }
        LocalFunction::parse(
            module,
            &self.indices,
            self.id,
            self.ty,
            self.args.clone(),
            body,
            None,
            &mut validator,
            &mut Vec::new(),
        )
    }

    pub(super) fn emit(
        &self,
        module: &Module,
        indices: &IdsToIndices,
        emit_local_names: bool,
        scratch: &mut EmitScratch,
    ) -> (Vec<u8>, IdHashMap<Local, u32>) {
        let original_locals = self.indices.function_locals(self.id);
        let base = original_locals.first().map_or(0, |id| id.index());
        debug_assert!(original_locals
            .iter()
            .enumerate()
            .all(|(i, id)| id.index() == base + i));
        scratch
            .local_indices
            .resize(original_locals.len(), u32::MAX);
        scratch.local_indices.fill(u32::MAX);
        for group in &mut scratch.numeric_groups {
            group.clear();
        }
        scratch.reference_groups.clear();
        for (index, slot) in scratch
            .local_indices
            .iter_mut()
            .take(self.args.len())
            .enumerate()
        {
            *slot = index as u32;
        }
        for &id in &self.facts.used_locals {
            let original = id.index() - base;
            if original < self.args.len() {
                continue;
            }
            let group = match module.locals.get(id).ty() {
                ValType::I32 => &mut scratch.numeric_groups[0],
                ValType::I64 => &mut scratch.numeric_groups[1],
                ValType::F32 => &mut scratch.numeric_groups[2],
                ValType::F64 => &mut scratch.numeric_groups[3],
                ValType::V128 => &mut scratch.numeric_groups[4],
                ValType::Ref(ty) => scratch.reference_groups.entry(ty).or_default(),
            };
            group.push(original);
        }
        let mut next = self.args.len() as u32;
        let mut declarations = Vec::new();
        // This is ValType's ordering, matching the IR emitter's BTreeMap.
        for (ty, group) in [
            ValType::I32,
            ValType::I64,
            ValType::F32,
            ValType::F64,
            ValType::V128,
        ]
        .into_iter()
        .zip(&mut scratch.numeric_groups)
        {
            if group.is_empty() {
                continue;
            }
            group.sort_unstable();
            for &original in group.iter() {
                scratch.local_indices[original] = next;
                next += 1;
            }
            declarations.push((group.len() as u32, ty.to_wasmencoder_type(indices)));
        }
        for (&ty, group) in &mut scratch.reference_groups {
            group.sort_unstable();
            for &original in group.iter() {
                scratch.local_indices[original] = next;
                next += 1;
            }
            declarations.push((
                group.len() as u32,
                ValType::Ref(ty).to_wasmencoder_type(indices),
            ));
        }
        let mut locals = IdHashMap::default();
        if emit_local_names {
            for (&id, &emitted) in original_locals.iter().zip(&scratch.local_indices) {
                if emitted != u32::MAX {
                    locals.insert(id, emitted);
                }
            }
        }
        let mut control = std::mem::take(&mut scratch.control);
        control.reset();
        let mut function = wasm_encoder::Function::new(declarations).into_raw_body();
        // Remapping usually keeps or reduces the input size. Reserve its body
        // length once instead of repeatedly reallocating the instruction bytes.
        function.reserve(self.body.len().saturating_sub(function.len()));
        let remap = Remap {
            input: &self.indices,
            output: indices,
            local_indices: &scratch.local_indices,
            blocks: &self.blocks,
        };
        let body = FunctionBody::new(wasmparser::BinaryReader::new(&self.body, self.offset));
        let mut reader = body
            .get_binary_reader_for_operators()
            .expect("validated body");
        let mut visitor = EmitVisitor {
            remap,
            bytes: function,
            control,
        };
        while !reader.eof() {
            reader
                .visit_operator(&mut visitor)
                .expect("validated opcode");
        }
        reader
            .finish_expression(&visitor)
            .expect("validated function end");
        let EmitVisitor { bytes, control, .. } = visitor;
        scratch.control = control;
        (bytes, locals)
    }
}

pub(crate) struct Scan {
    pub supported: bool,
    pub calls: Vec<FunctionId>,
    refs: References,
    facts: EmitFacts,
}

macro_rules! supported_proposal {
    (mvp) => {
        true
    };
    (sign_extension) => {
        true
    };
    (saturating_float_to_int) => {
        true
    };
    (bulk_memory) => {
        true
    };
    (threads) => {
        true
    };
    (tail_call) => {
        true
    };
    ($proposal:ident) => {
        false
    };
}

pub(crate) fn eligible(
    module: &Module,
    bodies: &[Scan],
    ids: &[FunctionId],
    roots: Vec<FunctionId>,
) -> Vec<bool> {
    use crate::{ExportItem, FunctionKind};
    let mut protected = IdHashSet::default();
    for export in module.exports.iter() {
        if let ExportItem::Function(f) = export.item {
            protected.insert(f);
        }
    }
    protected.extend(module.start);
    let imports: IdHashSet<_> = module
        .funcs
        .iter()
        .filter_map(|f| matches!(f.kind, FunctionKind::Import(_)).then_some(f.id()))
        .collect();
    let positions: IdHashMap<Function, usize> = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, id)| (id, i))
        .collect();
    log::debug!(target: "walrus::encoded", "protected endpoints: {}, inspection roots: {}", protected.len(), roots.len());
    let mut pinned: IdHashSet<_> = roots.into_iter().collect();
    let root_imports: IdHashSet<_> = pinned.intersection(&imports).copied().collect();
    for (&id, body) in ids.iter().zip(bodies) {
        if body.calls.iter().any(|f| root_imports.contains(f)) {
            pinned.insert(id);
        }
    }
    let mut queue: Vec<_> = pinned.iter().copied().collect();
    while let Some(id) = queue.pop() {
        if let Some(&index) = positions.get(&id) {
            for &callee in &bodies[index].calls {
                if pinned.insert(callee) {
                    queue.push(callee);
                }
            }
        }
    }
    log::debug!(target: "walrus::encoded", "inspection closure: {}", pinned.len());
    let mut rewrite_targets = protected.clone();
    // Inspection-only callees stay editable, but their callers need no call
    // rewrite. Binding endpoints above, and direct import callers below, do.
    protected.extend(pinned);
    rewrite_targets.extend(imports.iter().copied());
    for (&id, body) in ids.iter().zip(bodies) {
        if body.calls.iter().any(|f| imports.contains(f)) {
            rewrite_targets.insert(id);
        }
    }
    log::debug!(target: "walrus::encoded", "rewrite targets: {}", rewrite_targets.len());
    ids.iter()
        .zip(bodies)
        .map(|(id, body)| {
            body.supported
                && !protected.contains(id)
                && !body.calls.iter().any(|f| rewrite_targets.contains(f))
        })
        .collect()
}

pub(crate) fn clone_validation_metadata(
    validator: &FuncToValidate<ValidatorResources>,
) -> FuncToValidate<ValidatorResources> {
    FuncToValidate {
        resources: validator.resources.clone(),
        index: validator.index,
        ty: validator.ty,
        features: validator.features,
    }
}

macro_rules! record_field {
    ($s:ident, $i:ident, $l:ident, $b:ident, function_index, $v:ident) => {
        $s.refs.funcs.insert($i.get_func(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, global_index, $v:ident) => {
        $s.refs.globals.insert($i.get_global(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, local_index, $v:ident) => {
        $s.facts.used_locals.insert(
            *$l.get(*$v as usize)
                .ok_or_else(|| anyhow::anyhow!("local index out of bounds"))?,
        );
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, type_index, $v:ident) => {
        $s.refs.types.insert($i.get_type(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, table_index, $v:ident) => {
        $s.refs.tables.insert($i.get_table(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, table, $v:ident) => {
        $s.refs.tables.insert($i.get_table(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, dst_table, $v:ident) => {
        $s.refs.tables.insert($i.get_table(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, src_table, $v:ident) => {
        $s.refs.tables.insert($i.get_table(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, memarg, $v:ident) => {
        $s.refs.memories.insert($i.get_memory($v.memory)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, mem, $v:ident) => {
        $s.refs.memories.insert($i.get_memory(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, dst_mem, $v:ident) => {
        $s.refs.memories.insert($i.get_memory(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, src_mem, $v:ident) => {
        $s.refs.memories.insert($i.get_memory(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, data_index, $v:ident) => {
        $s.refs.data.insert($i.get_data(*$v)?);
        $s.facts.uses_data_segments = true;
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, elem_index, $v:ident) => {
        $s.refs.elements.insert($i.get_element(*$v)?);
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, blockty, $v:ident) => {
        if let BlockType::FuncType(index) = $v {
            match $b
                .get(*index as usize)
                .ok_or_else(|| anyhow::anyhow!("block type out of bounds"))?
            {
                BlockEncoding::Multi(id) => {
                    $s.refs.types.insert(*id);
                }
                BlockEncoding::Result(ValType::Ref(_)) => {
                    $s.supported = false;
                }
                _ => {}
            }
        } else if matches!($v, BlockType::Type(wasmparser::ValType::Ref(_))) {
            $s.supported = false;
        }
    };
    ($s:ident, $i:ident, $l:ident, $b:ident, $field:ident, $v:ident) => {};
}

impl Scan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: &FunctionBody<'_>,
        indices: &IndicesToIds,
        id: FunctionId,
        blocks: &[BlockEncoding],
        module: &Module,
        ty: TypeId,
        validation: &FuncToValidate<ValidatorResources>,
        allocations: &mut FuncValidatorAllocations,
    ) -> Result<Self> {
        let mut validator =
            clone_validation_metadata(validation).into_validator(std::mem::take(allocations));
        let mut reader = body.get_binary_reader();
        validator.read_locals(&mut reader)?;
        let mut scan = Self {
            supported: true,
            calls: Vec::new(),
            refs: References::default(),
            facts: EmitFacts {
                size: 0,
                uses_data_segments: false,
                ref_funcs: Vec::new(),
                used_locals: IdHashSet::default(),
            },
        };
        let signature = module.types.get(ty);
        scan.supported = !signature
            .params()
            .iter()
            .chain(signature.results())
            .any(|t| matches!(t, ValType::Ref(_)));
        scan.refs.types.insert(
            module
                .types
                .find_for_function_entry(signature.results())
                .unwrap(),
        );
        let locals = indices.function_locals(id);
        let mut visitor = ScanVisitor {
            scan,
            control: Control::new(),
            validator,
            pos: 0,
            indices,
            locals,
            blocks,
        };
        while !reader.eof() {
            visitor.pos = reader.original_position();
            reader.visit_operator(&mut visitor)??;
        }
        reader.finish_expression(&visitor)?;
        *allocations = visitor.validator.into_allocations();
        Ok(visitor.scan)
    }
}

macro_rules! control_code {
    (Block) => {
        ControlCode::Block
    };
    (Loop) => {
        ControlCode::Loop
    };
    (If) => {
        ControlCode::If
    };
    (Else) => {
        ControlCode::Else
    };
    (End) => {
        ControlCode::End
    };
    (Nop) => {
        ControlCode::Nop
    };
    (Br) => {
        ControlCode::Unconditional
    };
    (BrTable) => {
        ControlCode::Unconditional
    };
    (Return) => {
        ControlCode::Unconditional
    };
    (Unreachable) => {
        ControlCode::Unconditional
    };
    ($op:ident) => {
        ControlCode::Other
    };
}
#[derive(Clone, Copy)]
enum ControlCode {
    Block,
    Loop,
    If,
    Else,
    End,
    Nop,
    Unconditional,
    Other,
}
#[derive(Clone, Copy)]
enum Kind {
    Entry,
    Block,
    Loop,
    If,
    Else,
}
struct Frame {
    kind: Kind,
    enabled: bool,
    unreachable: bool,
}
enum Step {
    Skip,
    Emit,
    EndIf,
}
#[derive(Default)]
struct Control(Vec<Frame>);
impl Control {
    fn new() -> Self {
        let mut control = Self::default();
        control.reset();
        control
    }
    fn reset(&mut self) {
        self.0.clear();
        self.0.push(Frame {
            kind: Kind::Entry,
            enabled: true,
            unreachable: false,
        });
    }
    #[inline(always)]
    fn step(&mut self, op: ControlCode) -> Step {
        let frame = self.0.last_mut().expect("valid operator framing");
        let enabled = frame.enabled && !frame.unreachable;
        match op {
            ControlCode::Block | ControlCode::Loop | ControlCode::If => {
                let kind = match op {
                    ControlCode::If => Kind::If,
                    ControlCode::Loop => Kind::Loop,
                    _ => Kind::Block,
                };
                self.0.push(Frame {
                    kind,
                    enabled,
                    unreachable: false,
                });
            }
            ControlCode::Else => {
                frame.kind = Kind::Else;
                frame.unreachable = false;
                return if frame.enabled {
                    Step::Emit
                } else {
                    Step::Skip
                };
            }
            ControlCode::End => {
                let frame = self.0.pop().unwrap();
                return if !frame.enabled {
                    Step::Skip
                } else if matches!(frame.kind, Kind::If) {
                    Step::EndIf
                } else {
                    Step::Emit
                };
            }
            ControlCode::Nop => return Step::Skip,
            ControlCode::Unconditional => {
                frame.unreachable = true;
            }
            ControlCode::Other => {}
        }
        if enabled {
            Step::Emit
        } else {
            Step::Skip
        }
    }
}

struct ScanVisitor<'a> {
    scan: Scan,
    control: Control,
    validator: FuncValidator<ValidatorResources>,
    pos: usize,
    indices: &'a IndicesToIds,
    locals: &'a [LocalId],
    blocks: &'a [BlockEncoding],
}
impl wasmparser::FrameStack for ScanVisitor<'_> {
    fn current_frame(&self) -> Option<wasmparser::FrameKind> {
        self.validator.get_control_frame(0).map(|f| f.kind)
    }
}
macro_rules! validate_opcode {
    (simd, $v:ident, $visit:ident $(, $arg:ident)*) => { $v.simd_visitor().unwrap().$visit($($arg.clone()),*) };
    (relaxed_simd, $v:ident, $visit:ident $(, $arg:ident)*) => { $v.simd_visitor().unwrap().$visit($($arg.clone()),*) };
    ($proposal:ident, $v:ident, $visit:ident $(, $arg:ident)*) => { $v.$visit($($arg.clone()),*) };
}
macro_rules! record_call {
    (Call, $s:ident, $i:ident, $function:ident) => {
        $s.calls.push($i.get_func($function)?);
    };
    (ReturnCall, $s:ident, $i:ident, $function:ident) => {
        $s.calls.push($i.get_func($function)?);
    };
    ($op:ident, $s:ident, $i:ident $(, $arg:ident)*) => {};
}
macro_rules! scan_opcode {
    ($(@$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
        $(
            #[allow(unused_variables, clippy::clone_on_copy)]
            fn $visit(&mut self $($(, $arg: $argty)*)?) -> Self::Output {
                let mut validator = self.validator.visitor(self.pos);
                validate_opcode!($proposal, validator, $visit $($(, $arg)*)?)?;
                let scan = &mut self.scan;
                let indices = self.indices;
                record_call!($op, scan, indices $($(, $arg)*)?);
                if !supported_proposal!($proposal) { scan.supported = false; }
                if !scan.supported { return Ok(()); }
                let code = control_code!($op);
                if matches!(self.control.step(code), Step::Skip) { return Ok(()); }
                if !matches!(code, ControlCode::End | ControlCode::Else) { scan.facts.size += 1; }
                let locals = self.locals;
                let blocks = self.blocks;
                $($( {
                    let $arg = &$arg;
                    record_field!(scan, indices, locals, blocks, $arg, $arg);
                } )*)?
                Ok(())
            }
        )*
    };
}
impl<'a> VisitOperator<'a> for ScanVisitor<'_> {
    type Output = Result<()>;
    fn simd_visitor(&mut self) -> Option<&mut dyn VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }
    wasmparser::for_each_visit_operator!(scan_opcode);
}
impl<'a> VisitSimdOperator<'a> for ScanVisitor<'_> {
    wasmparser::for_each_visit_simd_operator!(scan_opcode);
}

struct EmitVisitor<'a> {
    remap: Remap<'a>,
    bytes: Vec<u8>,
    control: Control,
}
impl wasmparser::FrameStack for EmitVisitor<'_> {
    fn current_frame(&self) -> Option<wasmparser::FrameKind> {
        self.control.0.last().map(|f| match f.kind {
            Kind::Entry | Kind::Block => wasmparser::FrameKind::Block,
            Kind::Loop => wasmparser::FrameKind::Loop,
            Kind::If => wasmparser::FrameKind::If,
            Kind::Else => wasmparser::FrameKind::Else,
        })
    }
}
macro_rules! emit_opcode {
    ($(@$proposal:ident $op:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
        $(
            fn $visit(&mut self $($(, $arg: $argty)*)?) {
                match self.control.step(control_code!($op)) {
                    Step::Skip => {},
                    Step::EndIf => {
                        wasm_encoder::Instruction::Else.encode(&mut self.bytes);
                        wasm_encoder::Instruction::End.encode(&mut self.bytes);
                    },
                    Step::Emit => {
                        let instruction = self.remap.instruction(Operator::$op $({ $($arg),* })?).expect("validated instruction");
                        instruction.encode(&mut self.bytes);
                    },
                }
            }
        )*
    };
}
impl<'a> VisitOperator<'a> for EmitVisitor<'_> {
    type Output = ();
    fn simd_visitor(&mut self) -> Option<&mut dyn VisitSimdOperator<'a, Output = Self::Output>> {
        Some(self)
    }
    wasmparser::for_each_visit_operator!(emit_opcode);
}
impl<'a> VisitSimdOperator<'a> for EmitVisitor<'_> {
    wasmparser::for_each_visit_simd_operator!(emit_opcode);
}

struct Remap<'a> {
    input: &'a IndicesToIds,
    output: &'a IdsToIndices,
    local_indices: &'a [u32],
    blocks: &'a [BlockEncoding],
}
macro_rules! remap_index {
    ($name:ident, $get:ident, $output:ident) => {
        fn $name(&mut self, index: u32) -> std::result::Result<u32, reencode::Error<Self::Error>> {
            Ok(self
                .output
                .$output(self.input.$get(index).expect("validated index")))
        }
    };
}
impl Reencode for Remap<'_> {
    type Error = Infallible;
    remap_index!(function_index, get_func, get_func_index);
    remap_index!(global_index, get_global, get_global_index);
    remap_index!(memory_index, get_memory, get_memory_index);
    remap_index!(table_index, get_table, get_table_index);
    remap_index!(type_index, get_type, get_type_index);
    remap_index!(data_index, get_data, get_data_index);
    remap_index!(element_index, get_element, get_element_index);
    remap_index!(tag_index, get_tag, get_tag_index);

    fn block_type(
        &mut self,
        ty: BlockType,
    ) -> std::result::Result<wasm_encoder::BlockType, reencode::Error<Self::Error>> {
        if let BlockType::FuncType(index) = ty {
            return Ok(match self.blocks[index as usize] {
                BlockEncoding::Empty => wasm_encoder::BlockType::Empty,
                BlockEncoding::Result(ty) => {
                    wasm_encoder::BlockType::Result(ty.to_wasmencoder_type(self.output))
                }
                BlockEncoding::Multi(id) => {
                    wasm_encoder::BlockType::FunctionType(self.output.get_type_index(id))
                }
            });
        }
        reencode::utils::block_type(self, ty)
    }

    #[inline(always)]
    fn instruction<'a>(
        &mut self,
        op: Operator<'a>,
    ) -> std::result::Result<wasm_encoder::Instruction<'a>, reencode::Error<Self::Error>> {
        let local = |index: u32| {
            let emitted = self.local_indices[index as usize];
            assert_ne!(emitted, u32::MAX, "local must have an emitted index");
            emitted
        };
        Ok(match op {
            Operator::LocalGet { local_index } => {
                wasm_encoder::Instruction::LocalGet(local(local_index))
            }
            Operator::LocalSet { local_index } => {
                wasm_encoder::Instruction::LocalSet(local(local_index))
            }
            Operator::LocalTee { local_index } => {
                wasm_encoder::Instruction::LocalTee(local(local_index))
            }
            Operator::Call { function_index } => {
                wasm_encoder::Instruction::Call(self.function_index(function_index)?)
            }
            Operator::GlobalGet { global_index } => {
                wasm_encoder::Instruction::GlobalGet(self.global_index(global_index)?)
            }
            Operator::GlobalSet { global_index } => {
                wasm_encoder::Instruction::GlobalSet(self.global_index(global_index)?)
            }
            Operator::I32Const { value } => wasm_encoder::Instruction::I32Const(value),
            Operator::I64Const { value } => wasm_encoder::Instruction::I64Const(value),
            Operator::F32Const { value } => wasm_encoder::Instruction::F32Const(value.into()),
            Operator::F64Const { value } => wasm_encoder::Instruction::F64Const(value.into()),
            Operator::I32Add => wasm_encoder::Instruction::I32Add,
            Operator::I32Sub => wasm_encoder::Instruction::I32Sub,
            Operator::I32Mul => wasm_encoder::Instruction::I32Mul,
            Operator::I32Eqz => wasm_encoder::Instruction::I32Eqz,
            Operator::I32Load { memarg } => {
                wasm_encoder::Instruction::I32Load(self.mem_arg(memarg)?)
            }
            Operator::I64Load { memarg } => {
                wasm_encoder::Instruction::I64Load(self.mem_arg(memarg)?)
            }
            Operator::F32Load { memarg } => {
                wasm_encoder::Instruction::F32Load(self.mem_arg(memarg)?)
            }
            Operator::F64Load { memarg } => {
                wasm_encoder::Instruction::F64Load(self.mem_arg(memarg)?)
            }
            Operator::I32Store { memarg } => {
                wasm_encoder::Instruction::I32Store(self.mem_arg(memarg)?)
            }
            Operator::I64Store { memarg } => {
                wasm_encoder::Instruction::I64Store(self.mem_arg(memarg)?)
            }
            Operator::F32Store { memarg } => {
                wasm_encoder::Instruction::F32Store(self.mem_arg(memarg)?)
            }
            Operator::F64Store { memarg } => {
                wasm_encoder::Instruction::F64Store(self.mem_arg(memarg)?)
            }
            other => return reencode::utils::instruction(self, other),
        })
    }
}
