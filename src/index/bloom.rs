use super::*;

/// Alias for the native implementation of a flat index.
pub type BinaryBloomIndex = BinaryBloomIndexImpl;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct BinaryBloomIndexImpl {
    inner: *mut FaissIndexBinaryBloom,
}

unsafe impl Send for BinaryBloomIndexImpl {}
unsafe impl Sync for BinaryBloomIndexImpl {}

impl Drop for BinaryBloomIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexBinaryBloom_free(self.inner);
        }
    }
}

impl BinaryBloomIndexImpl {
    pub fn new(d: u32) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexBinaryBloom_new(
                &mut inner,
                (d & 0x7FFF_FFFF) as idx_t,
            ))?;
            Ok(BinaryBloomIndexImpl { inner })
        }
    }

    pub fn d(&self) -> u32 {
        unsafe { faiss_IndexBinaryBloom_d(self.inner as *mut FaissIndexBinaryBloom) as u32 }
    }

    pub fn add(&mut self, x: &[u8]) -> Result<()> {
        unsafe {
            let n = x.len() / (self.d() / 8) as usize;
            faiss_try(faiss_IndexBinaryBloom_add(
                self.inner as *mut FaissIndexBinaryBloom,
                n as i64,
                x.as_ptr(),
            ))?;
            Ok(())
        }
    }

    pub fn reject(&mut self, query: &[u8]) -> Result<crate::index::RejectionResult> {
        unsafe {
            let d = self.d() as usize;
            let nq = query.len() / (d / 8);
            let mut p_rej: *mut FaissRejectionResult = ::std::ptr::null_mut();
            faiss_try(faiss_RejectionResult_new(&mut p_rej, nq))?;
            faiss_try(faiss_IndexBinaryBloom_reject(
                self.inner as *mut FaissIndexBinaryBloom,
                nq as idx_t,
                query.as_ptr(),
                p_rej,
            ))?;
            Ok(crate::index::RejectionResult { inner: p_rej })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: u32 = 256;

    #[test]
    fn binary_bloom_index() {
        let mut index = BinaryBloomIndexImpl::new(D).unwrap();
        let data = vec![0u8; { D as usize / 8 } * 2];
        index.add(&data).unwrap();

        let rejections = index.reject(&data).unwrap();
        assert_eq!(rejections.nq(), 2);
        assert_eq!(rejections.rejections(), [false, false]);

        let query = vec![1u8; D as usize / 8];
        let rejections = index.reject(&query).unwrap();
        assert_eq!(rejections.nq(), 1);
        assert_eq!(rejections.rejections(), [true]);
    }
}
