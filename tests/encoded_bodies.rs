use walrus::{FunctionKind, Module, ModuleConfig};

fn no_inspection_roots(_: &Module) -> Vec<walrus::FunctionId> {
    Vec::new()
}
fn encoded_config() -> ModuleConfig {
    let mut config = ModuleConfig::new();
    config
        .preserve_original_code_metadata(false)
        .preserve_encoded_bodies(Some(no_inspection_roots));
    config
}
fn compare(wat: &str, encoded: bool) {
    let input = wat::parse_str(wat).unwrap();
    let mut reference = ModuleConfig::new().parse(&input).unwrap();
    let mut candidate = encoded_config().parse(&input).unwrap();
    assert_eq!(
        candidate
            .funcs
            .iter()
            .any(|f| matches!(f.kind, FunctionKind::Encoded(_))),
        encoded
    );
    // Compare both before and after reachability pruning. This checks body
    // normalization, local/type/function index remapping, and retained roots.
    assert_eq!(
        reference.emit_wasm(),
        candidate.emit_wasm(),
        "before GC: {wat}"
    );
    walrus::passes::gc::run(&mut reference);
    walrus::passes::gc::run(&mut candidate);
    let output = candidate.emit_wasm();
    assert_eq!(reference.emit_wasm(), output, "after GC: {wat}");
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&output)
        .unwrap();
}

#[test]
fn encoded_scalar_and_control_bodies_match_instruction_ir() {
    for (signature, body, root) in [
        (
            "(param i32 i32) (result i32)",
            "local.get 0 i32.const 100 i32.mul local.get 1 i32.add",
            "i32.const 7 i32.const 3 call $helper drop",
        ),
        (
            "(param i32 i64 i32) (result i32)",
            "local.get 1 drop local.get 0 local.get 2 i32.sub",
            "i32.const 7 i64.const 9 i32.const 3 call $helper drop",
        ),
        (
            "(param i32) (result i32) (local f64 i32 i64 i32)",
            "i32.const 7 local.set 4 local.get 0 local.get 4 i32.add",
            "i32.const 3 call $helper drop",
        ),
        (
            "(param i32) (result i32)",
            "local.get 0 if (result i32) i32.const 7 else i32.const 9 end",
            "i32.const 1 call $helper drop",
        ),
        (
            "(param i32)",
            "local.get 0 if nop end",
            "i32.const 0 call $helper",
        ),
        (
            "(param i32) (result i32)",
            "local.get 0 return block i32.const 9 drop end i32.const 12",
            "i32.const 1 call $helper drop",
        ),
        (
            "(result i32)",
            "unreachable block loop br 0 end end i32.const 7",
            "call $helper drop",
        ),
        (
            "(param i32) (result i32)",
            "block local.get 0 br_table 0 0 end i32.const 7",
            "i32.const 0 call $helper drop",
        ),
        (
            "(result i32)",
            "block (result i32) i32.const 7 end",
            "call $helper drop",
        ),
        (
            "(result i32)",
            "i32.const 7 loop (param i32) (result i32) end",
            "call $helper drop",
        ),
    ] {
        compare(
            &format!("(module (func $helper {signature} {body}) (func (export \"root\") {root}))"),
            true,
        );
    }
}

#[test]
fn encoded_bodies_preserve_memory_table_global_and_type_roots() {
    compare(
        r#"(module
        (type $simple (func (result i32)))
        (memory $used 1) (memory $unused 1)
        (global $g (mut i32) (i32.const 7))
        (global $dead i32 (i32.const 9))
        (table $table 1 funcref)
        (data $data "abc") (data $unused_data "def")
        (elem $elem func $target)
        (func $target)
        (func $helper (result i32)
            i32.const 0 i32.const 0 i32.const 3 memory.init $data data.drop $data
            i32.const 3 i32.const 0 i32.const 3 memory.copy
            i32.const 6 i32.const 0 i32.const 3 memory.fill
            i32.const 0 i32.const 0 i32.const 1 table.init $elem elem.drop $elem
            block (type $simple) global.get $g end)
        (func (export "root") call $helper drop))"#,
        true,
    );
}

#[test]
fn encoded_atomic_and_tail_call_bodies_match_instruction_ir() {
    compare(
        r#"(module
        (memory 1 1 shared)
        (func $helper
            i32.const 0 i32.atomic.load drop
            i32.const 0 i32.const 1 i32.atomic.rmw.add drop
            i32.const 0 i32.const 0 i64.const 0 memory.atomic.wait32 drop
            i32.const 0 i64.const 0 i64.const 0 memory.atomic.wait64 drop
            i32.const 0 i32.const 0 memory.atomic.notify drop)
        (func (export "root") call $helper))"#,
        true,
    );
    compare(
        r#"(module
        (func $leaf (param i32) (result i32) local.get 0)
        (func $helper (param i32) (result i32) local.get 0 return_call $leaf i32.const 9)
        (func (export "root") i32.const 7 call $helper drop))"#,
        true,
    );
}

#[test]
fn import_callers_and_unsupported_opcodes_keep_ir() {
    compare(
        r#"(module (import "env" "f" (func $import))
        (func $helper call $import) (func (export "root") call $helper))"#,
        false,
    );
    compare(
        r#"(module (table 1 funcref) (elem (i32.const 0) $helper)
        (func $helper) (func (export "root") call $helper))"#,
        true,
    );
    compare(
        r#"(module (func $helper v128.const i32x4 1 2 3 4 drop)
        (func (export "root") call $helper))"#,
        false,
    );
}

#[test]
fn inspection_roots_pin_the_entire_callee_graph() {
    let input = wat::parse_str(
        r#"(module
        (import "env" "marker" (func $marker))
        (func $leaf (result i32) i32.const 7)
        (func $middle (result i32) call $leaf)
        (func $describe call $middle drop call $marker)
        (func (export "root") call $describe))"#,
    )
    .unwrap();
    let mut config = encoded_config();
    config.preserve_encoded_bodies(Some(|module| {
        module
            .imports
            .iter()
            .filter_map(|i| {
                if let walrus::ImportKind::Function(f) = i.kind {
                    Some(f)
                } else {
                    None
                }
            })
            .collect()
    }));
    let module = config.parse(&input).unwrap();
    assert!(module
        .funcs
        .iter()
        .all(|f| !matches!(f.kind, FunctionKind::Encoded(_))));
}

#[test]
fn materialization_restores_editable_ir_before_transforms() {
    let input = wat::parse_str(
        r#"(module (func $helper (result i32) i32.const 7)
        (func (export "root") call $helper drop))"#,
    )
    .unwrap();
    let mut module = encoded_config().parse(&input).unwrap();
    let id = module.funcs.by_name("helper").unwrap();
    assert!(matches!(
        module.funcs.get(id).kind,
        FunctionKind::Encoded(_)
    ));
    let before = module.emit_wasm();
    module.materialize_function(id).unwrap();
    assert!(matches!(module.funcs.get(id).kind, FunctionKind::Local(_)));
    assert_eq!(before, module.emit_wasm());
}

#[test]
fn original_metadata_requirements_override_encoded_opt_in() {
    let input =
        wat::parse_str(r#"(module (func $helper) (func (export "root") call $helper))"#).unwrap();
    let mut with_callback = encoded_config();
    with_callback.on_instr_loc(|p| walrus::ir::InstrLocId::new(*p as u32));
    for config in [
        ModuleConfig::new()
            .preserve_encoded_bodies(Some(no_inspection_roots))
            .clone(),
        encoded_config().preserve_code_transform(true).clone(),
        encoded_config().generate_dwarf(true).clone(),
        with_callback,
    ] {
        let module = config.parse(&input).unwrap();
        assert!(module
            .funcs
            .iter()
            .all(|f| !matches!(f.kind, FunctionKind::Encoded(_))));
    }
}

#[test]
fn encoded_validation_rejects_invalid_instruction_typing() {
    let input = wat::parse_str("(module (func (result i32) i64.const 0))").unwrap();
    assert!(encoded_config().parse(&input).is_err());
}

#[test]
fn encoded_locals_keep_concrete_heap_types_alive() {
    compare(
        r#"(module
        (type $live (struct (field i32)))
        (type $dead (struct (field i64)))
        (func $helper (local (ref null $live)) local.get 0 drop)
        (func (export "root") call $helper))"#,
        true,
    );
}

#[test]
fn encoded_indirect_calls_and_memory64_remap_indices_after_gc() {
    compare(
        r#"(module
        (type $unused (func (param f64)))
        (type $signature (func (param i32) (result i32)))
        (table 1 funcref)
        (elem (i32.const 0) $target)
        (func $target (type $signature) local.get 0)
        (func $helper (type $signature)
            local.get 0 i32.const 0 call_indirect (type $signature))
        (func (export "root") i32.const 7 call $helper drop))"#,
        true,
    );
    compare(
        r#"(module
        (memory $unused 1)
        (memory $used i64 1)
        (func $helper (param i64) (result i64) local.get 0 i64.load $used)
        (func (export "root") i64.const 0 call $helper drop))"#,
        true,
    );
}

#[test]
fn materializing_after_original_types_are_pruned_returns_an_error() {
    let input = wat::parse_str(
        r#"(module
        (type $unused (func (param i64) (result i64)))
        (func $helper (result i32) i32.const 7)
        (func (export "root") call $helper drop))"#,
    )
    .unwrap();
    let mut module = encoded_config().parse(&input).unwrap();
    let id = module.funcs.by_name("helper").unwrap();
    walrus::passes::gc::run(&mut module);
    assert!(module.materialize_function(id).is_err());
}

#[test]
fn named_arguments_and_locals_retain_their_emitted_indices() {
    compare(
        r#"(module
        (func $helper (param $first i32) (param $second i32) (result i32) (local $value i32)
            local.get $first local.set $value local.get $value local.get $second i32.sub)
        (func (export "root") i32.const 7 i32.const 3 call $helper drop))"#,
        true,
    );
}

#[test]
fn emission_scratch_resets_across_different_local_layouts() {
    compare(
        r#"(module
        (func $first (param i64 i32) (result i64) (local i32 i64 f32)
            local.get 1 local.set 2 local.get 0 local.set 3
            f32.const 7 local.set 4 local.get 3)
        (func $second (param i32) (result i32) (local f64)
            local.get 0)
        (func $third (result f64) (local f64 i32 f32)
            f64.const 9 local.set 0 local.get 0)
        (func (export "root")
            i64.const 7 i32.const 3 call $first drop
            i32.const 4 call $second drop call $third drop))"#,
        true,
    );
}

#[test]
fn reused_validators_reject_locals_from_a_previous_function() {
    for wat in [
        // No local 0 exists in the second definition.
        "(module (func (param i32) local.get 0 drop) (func local.get 0 drop))",
        // Local 0 exists, but now has a different type.
        "(module (func (param i64) local.get 0 i64.eqz drop) (func (param i32) local.get 0 i64.eqz drop))",
    ] {
        let input = wat::parse_str(wat).unwrap();
        for config in [ModuleConfig::new(), encoded_config()] {
            assert!(config.parse(&input).is_err(), "accepted {wat}");
        }
    }
}

#[test]
fn deferred_validation_retains_local_declaration_limits() {
    let mut module = wasm_encoder::Module::new();
    let mut types = wasm_encoder::TypeSection::new();
    types.ty().function([], []);
    module.section(&types);
    let mut functions = wasm_encoder::FunctionSection::new();
    functions.function(0).function(0);
    module.section(&functions);
    let mut code = wasm_encoder::CodeSection::new();
    let mut valid = wasm_encoder::Function::new([(2, wasm_encoder::ValType::I64)]);
    valid.instruction(&wasm_encoder::Instruction::End);
    code.function(&valid);
    let mut oversized = wasm_encoder::Function::new([(50_001, wasm_encoder::ValType::I32)]);
    oversized.instruction(&wasm_encoder::Instruction::End);
    code.function(&oversized);
    module.section(&code);
    for config in [ModuleConfig::new(), encoded_config()] {
        let error = config.parse(module.as_slice()).unwrap_err();
        assert!(format!("{error:#}").contains("too many locals"));
    }
}
