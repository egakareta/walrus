//! Walrus-specific hash maps and hash sets which typically hash much more
//! quickly than libstd's hash map which isn't optimized for speed.

use id_arena::Id;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// Type parameter for the hasher of a `HashMap` or `HashSet`
#[derive(Default, Copy, Clone, Debug)]
pub struct BuildIdHasher;

/// Hasher constructed by `BuildIdHasher`, only ever used to hash an `Id`
#[derive(Default, Copy, Clone, Debug)]
pub struct IdHasher {
    hash: u64,
}

pub type IdHashMap<K, V> = HashMap<Id<K>, V, BuildIdHasher>;
pub type IdHashSet<T> = HashSet<Id<T>, BuildIdHasher>;

impl BuildHasher for BuildIdHasher {
    type Hasher = IdHasher;

    fn build_hasher(&self) -> IdHasher {
        IdHasher { hash: 0 }
    }
}

// This is the "speed" of this hasher. We're only ever going to hash `Id<T>`,
// and `Id<T>` is composed of two parts: one part arena ID and one part index
// in the arena. The arena ID is a 32-bit integer and the index is a `usize`, so
// we're only going to handle those two types here.
//
// The goal is to produce a 64-bit result, and 32 of those comes from the arena
// ID. We can assume that none of the arena indexes are larger than 32-bit
// because wasm can't encode more than 2^32 items anyway, so we can basically
// just shift each item into the lower 32-bits of the hash. Ideally this'll end
// up producing a very optimized version of hashing!
//
// To explore this a bit, see https://godbolt.org/z/QxkXgF
impl Hasher for IdHasher {
    fn write_u32(&mut self, amt: u32) {
        self.hash <<= 32;
        self.hash |= amt as u64;
    }

    fn write_usize(&mut self, amt: usize) {
        self.hash <<= 32;
        self.hash |= amt as u64;
    }

    fn write(&mut self, _other: &[u8]) {
        panic!("hashing an `Id` should only be usize/u32")
    }

    fn finish(&self) -> u64 {
        // HashMap uses the high seven bits as a SIMD control-byte fingerprint.
        // Concatenating arena and slot leaves those bits identical for every
        // entry in an arena. Mix the sequential slot into the whole hash so
        // lookups can reject unrelated entries without comparing their IDs.
        self.hash.wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_slots_have_distinct_control_byte_fingerprints() {
        let fingerprints: HashSet<_> = (0..1024)
            .map(|index| {
                let mut hash = IdHasher::default();
                hash.write_u32(1);
                hash.write_usize(index);
                hash.finish() >> 57
            })
            .collect();
        assert!(fingerprints.len() > 100);
    }

    #[test]
    fn lookups_removals_and_other_arenas_are_distinguished() {
        let mut first: id_arena::Arena<usize> = id_arena::Arena::new();
        let mut second: id_arena::Arena<usize> = id_arena::Arena::new();
        let mut map = IdHashMap::default();
        let ids: Vec<_> = (0..4096)
            .map(|value| {
                let id = first.alloc(value);
                let other = second.alloc(value);
                map.insert(id, value);
                assert!(!map.contains_key(&other));
                id
            })
            .collect();
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(map.get(id), Some(&index));
            if index % 3 == 0 {
                assert_eq!(map.remove(id), Some(index));
            }
        }
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(map.contains_key(id), index % 3 != 0);
        }
    }
}
