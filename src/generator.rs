use ndarray::prelude::*;

use std::{collections::HashMap, f64};

use crate::types::{Algorithm, AlgorithmFunc, NonNanF64};

pub(super) fn word_to_vector() -> HashMap<String, Vec<f64>> {
    let words = std::fs::read_to_string("test_json.json").unwrap();

    let words_to_vec: HashMap<String, Vec<f64>> = serde_json::from_str(&words).unwrap();

    words_to_vec
}

trait KNearestN {
    // add code here

    /// Returns similar n vectors based on an algorithm specified and a return type(MinHeap or
    /// MaxHeap
    // for now we use a list
    fn find_similar_n<'a>(
        &'a self,
        search_vector: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
        search_list: impl Iterator<Item = &'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>>,
        algorithm: &'a Algorithm,
        n: usize,
    ) -> Vec<(&'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let mut heap_type = algorithm.init_heap(n);

        let algorithm_function: AlgorithmFunc = algorithm.into();

        for second_vector in search_list {
            let similarity = algorithm_function(search_vector, &second_vector);
            let heap_value: NonNanF64 = (second_vector, similarity).into();
            heap_type.push(heap_value)
        }

        heap_type.output()
    }
}

#[cfg(test)]
struct TestStore;

#[cfg(test)]
impl TestStore {
    fn new() -> Self {
        Self
    }
}

#[cfg(test)]
impl KNearestN for TestStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{MOST_SIMILAR, SEACH_TEXT, SENTENCES};

    #[test]
    fn test_find_top_3_similar_words_using_cosine_similarity_min_heap() {
        let sentences_vectors = word_to_vector();
        let mut heap = crate::types::MinHeap::new(3);
        let mut most_similar_vec: Vec<(&ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> =
            vec![];

        let first_vector =
            ndarray::Array1::<f64>::from_vec(sentences_vectors.get(SEACH_TEXT).unwrap().to_owned());
        let second_vector = ndarray::Array1::<f64>::from_vec(
            sentences_vectors.get(MOST_SIMILAR[0]).unwrap().to_owned(),
        );
        let similarity = crate::cosine_similarity(&first_vector, &second_vector);
        most_similar_vec.push((&second_vector, similarity));

        let heap_val: NonNanF64 = (&second_vector, similarity).into();

        heap.push(heap_val);

        let third_vector = ndarray::Array1::<f64>::from_vec(
            sentences_vectors.get(MOST_SIMILAR[1]).unwrap().to_owned(),
        );
        let similarity = crate::cosine_similarity(&first_vector, &third_vector);
        most_similar_vec.push((&third_vector, similarity));

        let heap_val: NonNanF64 = (&third_vector, similarity).into();
        heap.push(heap_val);
        assert_eq!(most_similar_vec.len(), heap.len());

        while let Some(value) = heap.pop() {
            let vector_value: (&ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64) =
                value.into();

            assert!(most_similar_vec.contains(&vector_value))
        }
    }
}
