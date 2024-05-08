use ndarray::prelude::*;
use std::cmp::Reverse;

use std::collections::BinaryHeap;

#[derive(PartialEq)]
pub(crate) struct NonNanF64<'a>((&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64));

impl<'a> From<(&ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> for NonNanF64<'a> {
    fn from(value: (&ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)) -> Self {
        value.into()
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
    heap: BinaryHeap<Reverse<NonNanF64<'a>>>,
}

impl<'a> MinHeap<'a> {
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
    pub(crate) fn push(&mut self, item: NonNanF64<'a>) {
        self.heap.push(Reverse(item));
    }
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

        while count <= n {
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
    heap: BinaryHeap<NonNanF64<'a>>,
}

impl<'a> MaxHeap<'a> {
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }
    fn push(&mut self, item: NonNanF64<'a>) {
        self.heap.push(item);
    }
    fn pop(&mut self) -> Option<NonNanF64<'a>> {
        self.heap.pop()
    }

    fn get_n(&mut self, n: usize) -> Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut output = vec![];
        let mut count: usize = 0;

        while count <= n {
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
