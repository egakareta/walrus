use gimli::{Reader, Section};
use std::borrow::Cow;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use walrus::{ir::InstrLocId, ModuleConfig};
use wasm_encoder::Encode;

fn input() -> Vec<u8> {
    let mut bytes =
        wat::parse_str("(module (func (export \"f\") (result i32) i32.const 42))").unwrap();
    bytes.push(0);
    wasm_encoder::CustomSection {
        name: Cow::Borrowed(".debug_str"),
        data: Cow::Borrowed(b"debug-name\0"),
    }
    .encode(&mut bytes);
    bytes
}

fn assert_mapping(config: &ModuleConfig, present: bool) {
    let module = config.parse(&input()).unwrap();
    let (_, function) = module.funcs.iter_local().next().unwrap();
    assert_eq!(!function.instruction_mapping.is_empty(), present);
    assert_eq!(function.original_range.is_some(), present);
    assert_eq!(
        !module
            .debug
            .dwarf
            .debug_str
            .borrow(|bytes| gimli::EndianSlice::new(bytes, gimli::LittleEndian))
            .reader()
            .is_empty(),
        present
    );
}

#[test]
fn original_metadata_is_preserved_by_default_and_optional_when_unused() {
    assert_mapping(&ModuleConfig::new(), true);
    let mut config = ModuleConfig::new();
    config.preserve_original_code_metadata(false);
    assert_mapping(&config, false);
    assert_mapping(&config.clone(), false);
    assert_eq!(
        ModuleConfig::new().parse(&input()).unwrap().emit_wasm(),
        config.parse(&input()).unwrap().emit_wasm()
    );
}

#[test]
fn dwarf_and_code_transforms_override_metadata_opt_out() {
    for dwarf in [true, false] {
        let mut config = ModuleConfig::new();
        config.preserve_original_code_metadata(false);
        if dwarf {
            config.generate_dwarf(true);
        } else {
            config.preserve_code_transform(true);
        }
        assert_mapping(&config, true);
    }
}

#[test]
fn location_callback_overrides_metadata_opt_out() {
    let count = Arc::new(AtomicUsize::new(0));
    let calls = count.clone();
    let mut config = ModuleConfig::new();
    config
        .preserve_original_code_metadata(false)
        .on_instr_loc(move |offset| {
            calls.fetch_add(1, Ordering::Relaxed);
            InstrLocId::new(*offset as u32)
        });
    assert_mapping(&config, true);
    assert!(count.load(Ordering::Relaxed) > 0);
}

#[test]
fn opting_out_does_not_disable_instruction_validation() {
    let invalid = wat::parse_str("(module (func (result i32) i64.const 0))").unwrap();
    for preserve in [true, false] {
        assert!(ModuleConfig::new()
            .preserve_original_code_metadata(preserve)
            .parse(&invalid)
            .is_err());
    }
}

#[test]
fn shared_control_signatures_round_trip_with_validation() {
    // Parameterized blocks/loops, explicit and implicit else, modern and
    // legacy EH (including multi-result catch handlers), and unreachable code.
    let cases = [
        "(module (func (export \"f\") (result i32 i64) i32.const 7 block (param i32) (result i32 i64) i64.const 9 end))",
        "(module (func (export \"f\") (param i32) (result i32) local.get 0 loop (param i32) (result i32) end))",
        "(module (func (export \"f\") (result i32) i32.const 7 i32.const 1 if (param i32) (result i32) i32.const 2 i32.add else i32.const 3 i32.add end))",
        "(module (func (export \"f\") (result i32) i32.const 7 i32.const 1 if (param i32) (result i32) end))",
        "(module (func (export \"f\") i32.const 1 if nop end))",
        "(module (tag $e (param i32)) (func (export \"f\") (result i32) try (result i32) i32.const 7 catch $e drop i32.const 8 catch_all i32.const 9 end))",
        "(module (tag $e (param i32)) (func (export \"f\") (result i32 i64) try (result i32 i64) i32.const 7 i64.const 9 catch $e drop i32.const 8 i64.const 10 catch_all i32.const 9 i64.const 11 end))",
        "(module (func (export \"f\") block $caught try_table (catch_all $caught) nop end end))",
        "(module (func (export \"f\") unreachable block i32.const 7 drop end))",
    ];
    for wat in cases {
        let input = wat::parse_str(wat).unwrap();
        let mut reference = ModuleConfig::new().parse(&input).unwrap();
        let mut lean = ModuleConfig::new()
            .preserve_original_code_metadata(false)
            .parse(&input)
            .unwrap();
        let output = lean.emit_wasm();
        wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
            .validate_all(&output)
            .unwrap();
        assert_eq!(reference.emit_wasm(), output);
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_instruction_layout_stays_within_budget() {
    assert_eq!(std::mem::size_of::<walrus::FunctionId>(), 8);
    assert!(std::mem::size_of::<walrus::ir::Instr>() <= 48);
    assert!(std::mem::size_of::<(walrus::ir::Instr, InstrLocId)>() <= 64);
}

#[test]
fn gc_preserves_types_referenced_only_by_used_locals() {
    let bytes = wat::parse_str(
        r#"(module
        (type $live (struct (field i32)))
        (type $dead (struct (field i64)))
        (func (export "f") (local (ref null $live)) local.get 0 drop)
    )"#,
    )
    .unwrap();
    let mut module = ModuleConfig::new().parse(&bytes).unwrap();
    walrus::passes::gc::run(&mut module);
    assert_eq!(
        module
            .types
            .iter()
            .filter(|ty| ty.as_struct().is_some())
            .count(),
        1
    );
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&module.emit_wasm())
        .unwrap();
}

#[test]
fn gc_numeric_locals_do_not_keep_unused_gc_types() {
    let bytes = wat::parse_str(
        r#"(module
        (type $dead (struct (field i32)))
        (func (export "f") (local i32) local.get 0 drop)
    )"#,
    )
    .unwrap();
    let mut module = ModuleConfig::new().parse(&bytes).unwrap();
    walrus::passes::gc::run(&mut module);
    assert_eq!(
        module
            .types
            .iter()
            .filter(|ty| ty.as_struct().is_some())
            .count(),
        0
    );
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&module.emit_wasm())
        .unwrap();
}

#[test]
fn direct_decoder_preserves_opcode_locations_and_validates_simd() {
    let input = wat::parse_str(
        r#"(module
        (memory 1)
        (func (export "f") (param i32) (result i32) (local i32)
            local.get 0 local.tee 1 drop
            block
                local.get 1 br_if 0
                i32.const 0 i32.const 42 i32.store
            end
            v128.const i32x4 1 2 3 4
            v128.const i32x4 5 6 7 8
            i32x4.add i32x4.extract_lane 0
            local.get 1 i32.add))"#,
    )
    .unwrap();
    let mut expected = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&input) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut reader = body.get_operators_reader().unwrap();
            while !reader.eof() {
                expected.push(reader.original_position());
                reader.read().unwrap();
            }
        }
    }
    let positions = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = positions.clone();
    let mut module = ModuleConfig::new()
        .on_instr_loc(move |pos| {
            recorded.lock().unwrap().push(*pos);
            InstrLocId::new(*pos as u32)
        })
        .parse(&input)
        .unwrap();
    assert_eq!(*positions.lock().unwrap(), expected);
    wasmparser::Validator::new_with_features(wasmparser::WasmFeatures::all())
        .validate_all(&module.emit_wasm())
        .unwrap();
}

#[test]
fn direct_decoder_rejects_malformed_bodies_without_panicking() {
    // Each body starts with the locals vector. Exercise decoder framing as
    // well as type/index validation, including an invalid SIMD lane.
    let bodies: &[&[u8]] = &[
        &[0, 0x41, 0x80],          // truncated signed LEB
        &[0, 0x01],                // missing function end
        &[0, 0x0b, 0x0b],          // instructions after function end
        &[0, 0x05, 0x0b],          // else without if
        &[0, 0x02, 0x40, 0x0b],    // unclosed outer function
        &[0, 0x20, 0, 0x0b],       // nonexistent local
        &[0, 0x0c, 1, 0x0b],       // nonexistent label
        &[0, 0x41, 0, 0x0b],       // unexpected result
        &[0, 0xfd, 0x1b, 4, 0x0b], // invalid i32x4.extract_lane
    ];
    for &body in bodies {
        let mut input = wasm_encoder::Module::new();
        let mut types = wasm_encoder::TypeSection::new();
        types.ty().function([], []);
        input.section(&types);
        let mut funcs = wasm_encoder::FunctionSection::new();
        funcs.function(0);
        input.section(&funcs);
        let mut code = wasm_encoder::CodeSection::new();
        code.raw(body);
        input.section(&code);
        for strict in [true, false] {
            assert!(
                ModuleConfig::new()
                    .strict_validate(strict)
                    .preserve_original_code_metadata(false)
                    .parse(input.as_slice())
                    .is_err(),
                "accepted {body:?}"
            );
        }
    }
}

#[test]
fn raw_function_emission_preserves_code_ranges_across_leb_sizes() {
    #[derive(Debug)]
    struct Capture(Arc<std::sync::Mutex<walrus::CodeTransform>>);
    impl walrus::CustomSection for Capture {
        fn name(&self) -> &str {
            "offset-test"
        }
        fn data(&self, _: &walrus::IdsToIndices) -> Cow<'_, [u8]> {
            Cow::Borrowed(&[])
        }
        fn apply_code_transform(&mut self, t: &walrus::CodeTransform) {
            *self.0.lock().unwrap() = walrus::CodeTransform {
                instruction_map: t.instruction_map.clone(),
                code_section_start: t.code_section_start,
                function_ranges: t.function_ranges.clone(),
            };
        }
    }
    let mut module =
        walrus::Module::with_config(ModuleConfig::new().preserve_code_transform(true).clone());
    for count in [1, 100, 6000] {
        let mut builder = walrus::FunctionBuilder::new(&mut module.types, &[], &[]);
        let mut body = builder.func_body();
        for _ in 0..count {
            body.i32_const(0).drop();
        }
        builder.finish(vec![], &mut module.funcs);
    }
    let transform = Arc::new(std::sync::Mutex::new(walrus::CodeTransform::default()));
    module.customs.add(Capture(transform.clone()));
    let output = module.emit_wasm();
    wasmparser::Validator::new().validate_all(&output).unwrap();
    let mut expected = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&output) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let range = body.range();
            let length = range.end - range.start;
            let mut prefix = Vec::new();
            length.encode(&mut prefix);
            expected.push(range.start - prefix.len()..range.end);
        }
    }
    let mut actual: Vec<_> = transform
        .lock()
        .unwrap()
        .function_ranges
        .iter()
        .map(|(_, range)| range.clone())
        .collect();
    actual.sort_by_key(|range| range.start);
    assert_eq!(actual, expected);
}

#[test]
fn argument_indices_retain_their_abi_order() {
    let input = wat::parse_str(
        r#"(module
        (func (export "f") (param i32 i64 i32) (result i32)
            local.get 1 drop local.get 0 local.get 2 i32.sub))"#,
    )
    .unwrap();
    let output = ModuleConfig::new().parse(&input).unwrap().emit_wasm();
    let mut locals = Vec::new();
    for payload in wasmparser::Parser::new(0).parse_all(&output) {
        if let wasmparser::Payload::CodeSectionEntry(body) = payload.unwrap() {
            let mut reader = body.get_operators_reader().unwrap();
            while !reader.eof() {
                if let wasmparser::Operator::LocalGet { local_index } = reader.read().unwrap() {
                    locals.push(local_index);
                }
            }
        }
    }
    assert_eq!(locals, [1, 0, 2]);
}
