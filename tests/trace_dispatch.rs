use std::sync::atomic::{AtomicUsize, Ordering};

struct TraceLog {
    parsing: AtomicUsize,
    traversal: AtomicUsize,
}

impl log::Log for TraceLog {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        let message = record.args().to_string();
        if message.starts_with("validate instruction:") {
            self.parsing.fetch_add(1, Ordering::Relaxed);
        } else if message.starts_with("dfs_in_order:") {
            self.traversal.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn flush(&self) {}
}

#[test]
fn trace_dispatch_preserves_logging_and_generated_wasm() {
    // A separate test binary isolates the process-global logging configuration.
    static LOGGER: TraceLog = TraceLog {
        parsing: AtomicUsize::new(0),
        traversal: AtomicUsize::new(0),
    };
    log::set_logger(&LOGGER).unwrap();
    let input = wat::parse_str("(module (func $helper (result i32) i32.const 7) (func (export \"f\") (result i32) call $helper))").unwrap();
    log::set_max_level(log::LevelFilter::Trace);
    let mut config = walrus::ModuleConfig::new();
    config
        .preserve_original_code_metadata(false)
        .preserve_encoded_bodies(Some(|_| Vec::new()));
    let trace_output = config.parse(&input).unwrap().emit_wasm();
    let parsing = LOGGER.parsing.load(Ordering::Relaxed);
    let traversal = LOGGER.traversal.load(Ordering::Relaxed);
    assert_eq!(parsing, 4); // helper constant/end, exported call/end
    assert!(traversal > 0);
    log::set_max_level(log::LevelFilter::Off);
    let quiet_output = config.parse(&input).unwrap().emit_wasm();
    assert_eq!(parsing, LOGGER.parsing.load(Ordering::Relaxed));
    assert_eq!(traversal, LOGGER.traversal.load(Ordering::Relaxed));
    assert_eq!(trace_output, quiet_output);
}
