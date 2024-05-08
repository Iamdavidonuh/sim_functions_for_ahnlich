use ndarray::prelude::*;
use std::cmp::Reverse;

use std::collections::BinaryHeap;

/// Implementations for MinHeap and MaxHeap was changed slightly.
/// We only want to create a Heap with a max_capacity set by N
///
/// For MaxHeap, use the Reverse to pop in the opposite direction, which is the
/// Smallest items first.
///
/// For MinHeap, we pop the largest items first, leaving only smaller values
///
/// Essentially the MaxHeap works like the Minheap and the MinHeap works like the maxHeap.
/// Since we only care about the Pop operation, this works for us
///
/// Ordering does not matter to us anymore, we just want to keep the most and the Least items
/// in the heap at all times without growing it exponentially.
///
/// This makes it  easier to return most/least N without growing both heaps to K
///

#[derive(PartialEq)]
pub(crate) struct NonNanF64<'a>((&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64));

impl<'a> From<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> for NonNanF64<'a> {
    fn from(value: (&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)) -> NonNanF64<'a> {
        NonNanF64((value.0, value.1))
    }
}

impl<'a> Eq for NonNanF64<'a> {}

impl PartialOrd for NonNanF64<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (other.0).1.partial_cmp(&(&self.0).1)
    }
}

impl Ord for NonNanF64<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        ((self.0).1).partial_cmp(&(other.0).1).unwrap()
    }
}

pub(crate) struct MinHeap<'a> {
    max_capacity: usize,
    heap: BinaryHeap<NonNanF64<'a>>,
}

impl<'a> MinHeap<'a> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(n),
            max_capacity: n,
        }
    }
    //pub(crate) fn push(&mut self, item: NonNanF64<'a>) {
    //    self.heap.push(Reverse(item));
    // }
    pub(crate) fn push(&mut self, item: NonNanF64<'a>) {
        if self.heap.len() < self.max_capacity {
            self.heap.push(item);
        } else if let Some(current_max) = self.heap.peek() {
            if item < *current_max {
                self.heap.pop();
                self.heap.push(item);
            }
        }
    }
    pub(crate) fn pop(&mut self) -> Option<NonNanF64<'a>> {
        self.heap.pop()
    }

    fn get_n(&mut self, n: usize) -> Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut output = vec![];
        let mut count: usize = 0;

        while count < n {
            match self.pop() {
                Some(value) => {
                    let vector = value.0 .0.to_owned();
                    let similarity = value.0 .1;

                    output.push((vector, similarity));
                }
                None => break,
            }
            count += 1
        }
        output
    }
}

pub(crate) struct MaxHeap<'a> {
    max_capacity: usize,
    heap: BinaryHeap<Reverse<NonNanF64<'a>>>,
}

impl<'a> MaxHeap<'a> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(n),
            max_capacity: n,
        }
    }
    fn push(&mut self, item: NonNanF64<'a>) {
        if self.heap.len() < self.max_capacity {
            self.heap.push(Reverse(item));
        } else if let Some(Reverse(current_min)) = self.heap.peek() {
            if item > *current_min {
                self.heap.pop();
                self.heap.push(Reverse(item));
            }
        }
    }
    //fn pop(&mut self) -> Option<NonNanF64<'a>> {
    //    self.heap.pop()
    //}
    fn pop(&mut self) -> Option<NonNanF64> {
        if let Some(reversed) = self.heap.pop() {
            Some(reversed.0)
        } else {
            None
        }
    }

    fn get_n(&mut self, n: usize) -> Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut output = vec![];
        let mut count: usize = 0;

        while count < n {
            match self.pop() {
                Some(value) => {
                    let vector = value.0 .0.to_owned();
                    let similarity = value.0 .1;

                    output.push((vector, similarity));
                }
                None => break,
            }
            count += 1
        }
        output
    }
}

pub(crate) enum AlgorithmHeapType<'a> {
    MIN(MinHeap<'a>),
    MAX(MaxHeap<'a>),
}

impl<'a> AlgorithmHeapType<'a> {
    pub(crate) fn push(&mut self, item: NonNanF64<'a>) {
        match self {
            Self::MAX(h) => h.push(item),
            Self::MIN(h) => h.push(item),
        }
    }

    pub(crate) fn pop(&mut self) {
        match self {
            Self::MIN(h) => h.pop(),
            Self::MAX(h) => h.pop(),
        };
    }

    pub(crate) fn get_max_n(
        &mut self,
        n: usize,
    ) -> Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        match self {
            Self::MIN(h) => h.get_n(n),
            Self::MAX(h) => h.get_n(n),
        }
    }
}

pub(crate) enum Algorithm {
    DotProduct,
    Euclidean,
    Cosine,
}

#[cfg(test)]
mod tests {
    use crate::types::{MaxHeap, MinHeap};

    use super::NonNanF64;

    #[test]
    fn test_max_heap_doesnt_grow_beyond_max_capacity() {
        let mut heap = MaxHeap::new(2);
        let mut count = 4;
        let first_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);

        while count > 0 {
            let similarity: f64 = 0.1 / count as f64;

            let item: NonNanF64 = (&first_vector, similarity).into();
            heap.push(item);

            count -= 1;
        }

        assert!(heap.heap.len() == 2);
    }
    #[test]
    fn test_min_heap_doesnt_grow_beyond_max_capacity() {
        let mut heap = MinHeap::new(2);
        let mut count = 4;
        let first_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);

        while count > 0 {
            let similarity: f64 = 0.1 / count as f64;

            let item: NonNanF64 = (&first_vector, similarity).into();
            heap.push(item);

            count -= 1;
        }

        assert!(heap.heap.len() == 2);
    }
}
