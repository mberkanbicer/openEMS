//! 4-D field storage array (component N, i, j, k) with N-I-J-K ordering.
//! Mirrors `tools/arraylib/array_nijk.h` in the C++ codebase.

use crate::constants::FDTD_FLOAT;

/// A 4-dimensional array indexed as `(n, i, j, k)` where `n` is the field
/// component (0=x, 1=y, 2=z) and `i, j, k` are spatial mesh indices.
///
/// Memory layout is N-I-J-K: the `n` dimension has the largest stride.
pub struct ArrayNIJK<T> {
    data: Vec<T>,
    extent: [usize; 4], // [N, I, J, K]
    stride: [usize; 4],
}

impl<T> ArrayNIJK<T> {
    /// Return the linear index for a (n, i, j, k) tuple.
    pub fn linear_index(&self, tuple: [usize; 4]) -> usize {
        self.stride[0] * tuple[0]
            + self.stride[1] * tuple[1]
            + self.stride[2] * tuple[2]
            + self.stride[3] * tuple[3]
    }

    /// Return the extent array [N, I, J, K].
    pub fn extent(&self) -> [usize; 4] {
        self.extent
    }

    /// Raw pointer to the first element (read-only).
    ///
    /// Used by the threaded update kernels, which partition the `i` range;
    /// the caller must uphold the aliasing rules described at the call site.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Raw pointer to the first element (mutable).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Element strides in the flat buffer, `[n, i, j, k]`.
    pub fn strides(&self) -> [usize; 4] {
        self.stride
    }

    /// Return the number of components (N dimension, always 3).
        /// Number of stored values (all components).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Bytes held by the storage buffer.
    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn n_components(&self) -> usize {
        self.extent[0]
    }

    /// Return the I dimension size.
    pub fn ni(&self) -> usize {
        self.extent[1]
    }

    /// Return the J dimension size.
    pub fn nj(&self) -> usize {
        self.extent[2]
    }

    /// Return the K dimension size.
    pub fn nk(&self) -> usize {
        self.extent[3]
    }

    /// Immutable access: `array(n, i, j, k)`.
    pub fn get(&self, n: usize, i: usize, j: usize, k: usize) -> &T {
        &self.data[self.linear_index([n, i, j, k])]
    }

    /// Mutable access: `array(n, i, j, k)`.
    pub fn get_mut(&mut self, n: usize, i: usize, j: usize, k: usize) -> &mut T {
        let idx = self.linear_index([n, i, j, k]);
        &mut self.data[idx]
    }

    /// Convenience access with a position array `[i, j, k]`.
    pub fn get_pos(&self, n: usize, pos: [usize; 3]) -> &T {
        self.get(n, pos[0], pos[1], pos[2])
    }

    /// Mutable convenience access with a position array `[i, j, k]`.
    pub fn get_mut_pos(&mut self, n: usize, pos: [usize; 3]) -> &mut T {
        self.get_mut(n, pos[0], pos[1], pos[2])
    }
}

impl<T: Clone + Default> ArrayNIJK<T> {
    /// Create a new array with the given spatial extents (I, J, K).
    /// The N dimension is fixed at 3 (x, y, z components).
    pub fn new(_name: &str, extent: [usize; 3]) -> Self {
        let n = 3usize;
        let i = extent[0];
        let j = extent[1];
        let k = extent[2];
        let size = n * i * j * k;

        let extent4 = [n, i, j, k];
        let stride = [i * j * k, j * k, k, 1];

        Self {
            data: vec![T::default(); size],
            extent: extent4,
            stride,
        }
    }
}

// Convenience type alias for the FDTD field storage used throughout the solver.
pub type FieldArray = ArrayNIJK<FDTD_FLOAT>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation_and_access() {
        let arr = ArrayNIJK::<f32>::new("test", [4, 5, 6]);
        assert_eq!(arr.extent(), [3, 4, 5, 6]);
        assert_eq!(arr.n_components(), 3);
        assert_eq!(arr.ni(), 4);
        assert_eq!(arr.nj(), 5);
        assert_eq!(arr.nk(), 6);
    }

    #[test]
    fn test_get_set() {
        let mut arr = ArrayNIJK::<f32>::new("test", [4, 5, 6]);
        *arr.get_mut(0, 1, 2, 3) = 42.0;
        assert_eq!(*arr.get(0, 1, 2, 3), 42.0);
        assert_eq!(*arr.get_pos(1, [2, 3, 4]), 0.0);
    }

    #[test]
    fn test_linear_index() {
        let arr = ArrayNIJK::<f32>::new("test", [4, 5, 6]);
        // stride[0] = 4*5*6 = 120, stride[1] = 5*6 = 30, stride[2] = 6, stride[3] = 1
        assert_eq!(arr.linear_index([0, 0, 0, 0]), 0);
        assert_eq!(arr.linear_index([1, 0, 0, 0]), 120);
        assert_eq!(arr.linear_index([0, 1, 0, 0]), 30);
        assert_eq!(arr.linear_index([0, 0, 1, 0]), 6);
        assert_eq!(arr.linear_index([0, 0, 0, 1]), 1);
        assert_eq!(arr.linear_index([2, 3, 4, 5]), 2 * 120 + 3 * 30 + 4 * 6 + 5);
    }
}
