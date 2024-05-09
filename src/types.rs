use ndarray::prelude::*;
use std::cmp::Reverse;

use std::collections::BinaryHeap;

pub(crate) type AlgorithmFunc = fn(
    &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
    &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
) -> f64;

impl<'a> From<&'a Algorithm> for AlgorithmFunc {
    fn from(value: &'a Algorithm) -> AlgorithmFunc {
        match value {
            Algorithm::Cosine => crate::cosine_similarity,

            Algorithm::Euclidean => crate::euclidean_distance,

            Algorithm::DotProduct => crate::dot_product,
        }
    }
}
#[derive(PartialEq, Debug)]
pub(crate) struct NonNanF64<'a>((&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64));

impl<'a> From<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> for NonNanF64<'a> {
    fn from(value: (&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)) -> NonNanF64<'a> {
        NonNanF64((value.0, value.1))
    }
}
impl<'a> From<NonNanF64<'a>> for (&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64) {
    fn from(value: NonNanF64<'a>) -> (&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64) {
        (value.0 .0, value.0 .1)
    }
}

impl<'a> Eq for NonNanF64<'a> {}

impl PartialOrd for NonNanF64<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.0).1.partial_cmp(&(&other.0).1)
    }
}

impl Ord for NonNanF64<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0).1.partial_cmp(&(&other.0).1).unwrap()
    }
}

pub(crate) struct MinHeap<'a> {
    max_capacity: usize,
    heap: BinaryHeap<Reverse<NonNanF64<'a>>>,
}

impl<'a> MinHeap<'a> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            max_capacity: n,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }
    pub(crate) fn push(&mut self, item: NonNanF64<'a>) {
        self.heap.push(Reverse(item));
    }
    pub(crate) fn pop(&mut self) -> Option<NonNanF64<'a>> {
        if let Some(popped_item) = self.heap.pop() {
            Some(popped_item.0)
        } else {
            None
        }
    }

    pub(crate) fn output(
        &mut self,
    ) -> Vec<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut result: Vec<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> = vec![];

        loop {
            match self.pop() {
                Some(value) => {
                    if result.len() < self.max_capacity {
                        let vector_sim = value.0;
                        result.push((vector_sim.0, vector_sim.1));
                    }
                }
                None => break,
            }
        }
        result
    }
}

pub(crate) struct MaxHeap<'a> {
    max_capacity: usize,
    heap: BinaryHeap<NonNanF64<'a>>,
}

impl<'a> MaxHeap<'a> {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            max_capacity: n,
        }
    }
    fn push(&mut self, item: NonNanF64<'a>) {
        self.heap.push(item);
    }
    pub(crate) fn pop(&mut self) -> Option<NonNanF64<'a>> {
        self.heap.pop()
    }
    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    fn output(&mut self) -> Vec<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut output: Vec<_> = vec![];

        loop {
            match self.heap.pop() {
                Some(value) => {
                    if output.len() < self.max_capacity {
                        let vector_sim = value.0;
                        output.push((vector_sim.0, vector_sim.1));
                    }
                }
                None => break,
            }
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
    pub(crate) fn pop(&mut self) -> Option<NonNanF64<'a>> {
        match self {
            Self::MAX(h) => h.pop(),
            Self::MIN(h) => h.pop(),
        }
    }

    pub(crate) fn output(
        &mut self,
    ) -> Vec<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        match self {
            Self::MIN(h) => h.output(),
            Self::MAX(h) => h.output(),
        }
    }
}

pub(crate) enum Algorithm {
    DotProduct,
    Euclidean,
    Cosine,
}

impl Algorithm {
    pub(crate) fn init_heap(&self, capacity: usize) -> AlgorithmHeapType {
        match self {
            Self::Cosine | Self::Euclidean => AlgorithmHeapType::MIN(MinHeap::new(capacity)),
            Self::DotProduct => AlgorithmHeapType::MAX(MaxHeap::new(capacity)),
        }
    }
}

#[cfg(test)]
#[derive(PartialEq, Debug)]
pub(crate) struct TestNonNanF64(f64);

#[cfg(test)]
impl From<f64> for TestNonNanF64 {
    fn from(value: f64) -> TestNonNanF64 {
        TestNonNanF64(value)
    }
}

#[cfg(test)]
impl Eq for TestNonNanF64 {}

#[cfg(test)]
impl PartialOrd for TestNonNanF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[cfg(test)]
impl Ord for TestNonNanF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}
#[cfg(test)]
mod tests {
    use crate::types::{MaxHeap, MinHeap, NonNanF64, Reverse};

    use super::*;

    #[test]
    fn test_b_heap() {
        let mut heap = std::collections::BinaryHeap::new();

        // Wrap values in `Reverse`
        let push_one: TestNonNanF64 = 1.0.into();
        let push_two: TestNonNanF64 = 2.0.into();
        let push_three: TestNonNanF64 = 5.0.into();

        heap.push(Reverse(push_one));
        heap.push(Reverse(push_three));
        heap.push(Reverse(push_two));

        // If we pop these scores now, they should come back in the reverse order.
        assert_eq!(heap.pop(), Some(Reverse(1.0.into())));
        assert_eq!(heap.pop(), Some(Reverse(2.0.into())));
        assert_eq!(heap.pop(), Some(Reverse(5.0.into())));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_min_heap_ordering_works() {
        let mut heap = MinHeap::new(3);
        let mut count = 0.0;
        let first_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);

        // If we pop these scores now, they should come back in the reverse order.
        while count < 5.0 {
            let similarity: f64 = 1.0 + count;
            let item: NonNanF64 = (&first_vector, similarity).into();

            heap.push(item);

            count += 1.0;
        }

        assert_eq!(heap.pop(), Some((&first_vector, 1.0).into()));
        assert_eq!(heap.pop(), Some((&first_vector, 2.0).into()));
        assert_eq!(heap.pop(), Some((&first_vector, 3.0).into()));
    }

    #[test]
    fn test_max_heap_ordering_works() {
        let mut heap = MaxHeap::new(3);
        let mut count = 0.0;
        let first_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);

        // If we pop these scores now, they should come back  the right order(max first).
        while count < 5.0 {
            let similarity: f64 = 1.0 + count;
            let item: NonNanF64 = (&first_vector, similarity).into();

            heap.push(item);

            count += 1.0;
        }

        assert_eq!(heap.pop(), Some((&first_vector, 5.0).into()));
        assert_eq!(heap.pop(), Some((&first_vector, 4.0).into()));
        assert_eq!(heap.pop(), Some((&first_vector, 3.0).into()));
    }
}
