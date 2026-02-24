use std::convert::TryInto;

/// A segment is essentially a bitmap of
/// representing values in [0, 65536).
///
///
/// # Examples
///
/// ```
/// use faiss::index::bloom::Segment;
///
/// let mut segment = Segment::empty();
/// segment.set(8000u16);
/// assert!(segment.get(8000u16), "should be able to fetch a value known to exist in the bitmap.");
/// ```
#[derive(Debug)]
pub struct Segment {
    data: Box<[u8; 8192]>,
}

impl Default for Segment {
    fn default() -> Self {
        Self {
            data: Box::new([0u8; 8192]),
        }
    }
}

impl Segment {
    /// Create an empty (yet initialized) segment.
    pub fn empty() -> Self {
        Default::default()
    }

    /// Get whether this value has been seen before.
    pub fn get(&self, index: u16) -> bool {
        let byte_offset = (index / 8) as usize;
        let bit_offset = 7 - ((index % 8) as usize);

        let byte = (*self.data)[byte_offset];
        byte & (1 << bit_offset) != 0
    }

    /// Set this value as seen.
    #[inline(always)]
    pub fn set(&mut self, index: u16) {
        let byte_offset = (index / 8) as usize;
        let bit_offset = 7 - ((index % 8) as usize);
        let byte = &mut (*self.data)[byte_offset];
        *byte |= 1 << bit_offset;
    }

    /// Get an iterator over all the positions that have been
    /// seen at least once.
    pub fn get_setbits(&self) -> impl Iterator<Item = u16> + use<'_> {
        (0..=65535u16).filter(move |&index| self.get(index))
    }
}

/// A Bloom filter like cache that
/// divides binary vectors into segments of 16 bits
/// and tracks whether segments at corresponding positions
/// have been seen before.
///
/// This cache can be used to quickly reject range queries
/// of a narrow radius since just by scanning the query vector
/// we can determine if enough segments have not been seen before
/// causing the query to fall outside that range from any vector
/// in a database.
///
///
/// # Examples
///
/// ```
/// use faiss::index::bloom::BloomCache;
///
/// let mut cache = BloomCache::new(4);
/// // 4 segments imply (4 x 16 = 64) bits (or 8 bytes) per vector.
///
/// let mut vectors = vec![0u8; 8 * 2];
/// vectors[8..16].copy_from_slice(&[255u8; 8]);
///
/// // Add 2 vectors into the cache.
/// // 0x0 and 0xFFFFFFFFFFFFFFFF
/// cache.add(&vectors);
///
/// // Since the vectors are present,
/// // a range query of distance 0 should not be rejected.
/// assert!(!cache.should_reject(&vectors[0..8], 0));
/// assert!(!cache.should_reject(&vectors[8..16], 0));
///
/// // Far enough vectors shouldn't be rejected anyway.
/// assert!(!cache.should_reject(&vectors[0..8], 100));
/// ```
#[derive(Debug)]
pub struct BloomCache {
    segments: Vec<Segment>,
    size: usize,
}

impl BloomCache {
    /// Create a new Cache with the given number of 16-bits wide segments.
    pub fn new(num_segments: usize) -> Self {
        Self {
            segments: (0..num_segments).map(|_| Segment::default()).collect(),
            size: 0,
        }
    }

    /// Add given vectors to the cache, marking the correspoding
    /// segments as seen.
    pub fn add(&mut self, vectors: &[u8]) {
        let indices = (0..self.segments.len()).cycle();
        let chunks = vectors.chunks_exact(2);

        for (chunk, segment_index) in chunks.zip(indices) {
            if let Some(segment) = self.segments.get_mut(segment_index) {
                let chunk: [u8; 2] = chunk.try_into().unwrap();
                segment.set(u16::from_be_bytes(chunk));
            }
        }
        self.size += 1;
    }

    /// Determine whether a range search at the given radius can be safely
    /// rejected (as no hits) for the given query vector.
    pub fn should_reject(&self, query: &[u8], acceptance_radius: i32) -> bool {
        if acceptance_radius as usize >= self.segments.len() {
            return false;
        }

        let chunks = query.chunks_exact(2);

        let mut misses: i32 = 0;

        for (chunk, segment) in chunks.zip(self.segments.iter()) {
            let chunk: [u8; 2] = chunk.try_into().unwrap();
            if !segment.get(u16::from_be_bytes(chunk)) {
                misses += 1;
                if misses > acceptance_radius {
                    return true;
                }
            }
        }
        false
    }

    /// Reset the cache and clear all indexed state.
    pub fn reset(&mut self) {
        self.segments.clear();
        self.size = 0;
    }

    /// Get the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Determine whether no vectors have been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get an iterator over the underlying segments.
    pub fn iter_segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter()
    }
    /// Get a mutable iterator over the underlying segments.
    pub fn iter_segments_mut(&mut self) -> impl Iterator<Item = &mut Segment> {
        self.segments.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rng, Rng};

    #[test]
    fn segment_ops() {
        let mut segment = Segment::default();
        let setbits = segment.get_setbits().collect::<Vec<_>>();
        assert!(setbits.is_empty());

        for idx in 0..=65535 {
            assert!(!segment.get(idx));
            segment.set(idx);
            assert!(segment.get(idx));
        }
        let setbits = segment.get_setbits().collect::<Vec<_>>();
        assert!(!setbits.is_empty());
        assert_eq!(setbits, (0..=65535u16).into_iter().collect::<Vec<_>>());
    }

    #[test]
    fn bloom_cache_recall() {
        let mut index = index_binary_factory(256, "BFlat")
            .unwrap()
            .into_flat()
            .unwrap();
        let mut cache = BloomCache::new(index.d() as usize / 16);

        let num_vectors: usize = 1000;
        let mut data = vec![0u8; (index.d() as usize / 8) * num_vectors];
        let mut rng = rng();
        rng.fill_bytes(&mut data);

        for vector in data.chunks_exact(32) {
            cache.add(vector);
            index.add(vector).unwrap();
        }

        for query in data.chunks_exact(32) {
            for radius in 1..16 {
                assert!(!cache.should_reject(query, radius));
                let hits = index.range_search(query, radius).unwrap();
                assert!(!hits.distances().is_empty());
            }
        }
    }

    #[test]
    fn bloom_cache_near() {
        let mut index = index_binary_factory(256, "BFlat")
            .unwrap()
            .into_flat()
            .unwrap();
        let mut cache = BloomCache::new(index.d() as usize / 16);

        let num_vectors: usize = 1;
        let mut data = vec![0u8; (index.d() as usize / 8) * num_vectors];
        let mut rng = rng();
        rng.fill_bytes(&mut data);

        for vector in data.chunks_exact(32) {
            cache.add(vector);
        }

        for corrupt_bit in (0..(index.d() as usize / 8)).step_by(2) {
            let mut query = data.clone();
            for idx in (0..(corrupt_bit + 1)).step_by(2) {
                query[idx] ^= 1 << 7;
            }
            for radius in 0..(corrupt_bit / 2 + 1) {
                assert!(cache.should_reject(&query, radius as i32))
            }
            for radius in (corrupt_bit / 2 + 1)..16 {
                assert!(!cache.should_reject(&query, radius as i32));
            }
            for idx in (0..(corrupt_bit + 1)).step_by(2) {
                query[idx] ^= 1 << 7;
            }
        }
    }
}
