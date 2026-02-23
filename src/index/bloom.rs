use std::iter::FromIterator;

use super::*;

#[derive(Debug)]
struct Segment {
    data: Box<[u8; 8192]>
}

impl Default for Segment {
    fn default() -> Self {
        Self {
            data: Box::new([0u8; 8192])
        }
    }
}

impl Segment {
    pub fn empty() -> Self {
        Default::default()
    }

    pub fn get(&self, index: u16) -> bool {
        let byte_offset = (index / 8) as usize;
        let bit_offset = 7 - ((index % 8) as usize);

        let byte = (*self.data)[byte_offset];
        (byte & (1 << bit_offset)) != 0
    }

    #[inline(always)]
    pub fn set(&mut self, index: u16) {
        let byte_offset = (index / 8) as usize;
        let bit_offset = 7 - ((index % 8) as usize);
        let byte = &mut (*self.data)[byte_offset];
        *byte |= 1 << bit_offset;
    }

    pub fn get_setbits(&self) -> impl Iterator<Item=u16> + use<'_> {
        (0..=65535u16)
        .filter(move |&index| self.get(index))
    }
}


#[derive(Debug)]
pub struct BloomCache {
    segments: Vec<Segment>
}

impl BloomCache {
    pub fn new(num_segments: usize) -> Self {
        Self { segments: (0..num_segments).into_iter().map(|_| Segment::default()).collect() }
    }

    pub fn add(&mut self, vectors: &[u8]) {
        let indices = (0..self.segments.len()).cycle();
        let (chunks, remainder) = vectors.as_chunks::<2>();
        let chunks_iter = chunks.iter();
        assert!(remainder.is_empty());

        for (chunk, segment_index) in chunks_iter.zip(indices) {
            if let Some(segment) = self.segments.get_mut(segment_index) {
                segment.set(u16::from_be_bytes(*chunk));
            }
        }
    }

    pub fn should_reject(&self, query: &[u8], acceptance_radius: i32) -> bool {
        if acceptance_radius as usize >= self.segments.len() {
            return false;
        }

        let (chunks, remainder) = query.as_chunks::<2>();
        assert!(remainder.is_empty());

        let mut misses: i32 = 0;

        for (chunk, segment) in chunks.iter().zip(self.segments.iter()) {
            if segment.get(u16::from_be_bytes(*chunk)) {
                misses += 1;
                if misses > acceptance_radius {
                    return true
                }
            }
        }
        false
    }

    pub fn reset(&mut self) {
        self.segments.clear();
    }
}



#[cfg(test)]
mod tests {
    use super::*;


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
    fn bloom_cache() {

        let mut index = index_binary_factory(256, "BFlat").unwrap().into_flat().unwrap();
        let mut cache = BloomCache::new(256 / 16);
    }
}