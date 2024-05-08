use ndarray::prelude::*;

use std::{collections::HashMap, f64};

use crate::types::{Algorithm, AlgorithmHeapType, MaxHeap, MinHeap, NonNanF64};

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
        &self,
        search_vector: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
        search_list: impl Iterator<Item = &'a ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>>,
        search_algorithm: &str,
        n: usize,
    ) -> Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, f64)> {
        let (algorithm, mut heap_type) = self.get_algorithm_and_heap_type(n, &search_algorithm);
        match algorithm {
            Algorithm::Cosine => {
                for second_vector in search_list {
                    let similarity = crate::cosine_similarity(search_vector, &second_vector);
                    let heap_value: NonNanF64 = (second_vector, similarity).into();

                    heap_type.push(heap_value)
                }
            }

            Algorithm::Euclidean => {
                for second_vector in search_list {
                    let similarity = crate::euclidean_distance(search_vector, &second_vector);
                    let heap_value: NonNanF64 = (second_vector, similarity).into();
                    heap_type.push(heap_value)
                }
            }

            Algorithm::DotProduct => {
                for second_vector in search_list {
                    let similarity = crate::dot_product(search_vector, &second_vector);
                    let heap_value: NonNanF64 = (second_vector, similarity).into();
                    heap_type.push(heap_value)
                }
            }
        }

        heap_type.get_max_n(n)
    }

    fn get_algorithm_and_heap_type(
        &self,
        heap_capacity: usize,
        search_algorithm: &str,
    ) -> (Algorithm, AlgorithmHeapType) {
        match search_algorithm {
            "cosine_similarity" => (
                Algorithm::Cosine,
                AlgorithmHeapType::MIN(MinHeap::new(heap_capacity)),
            ),
            "dot_product" => (
                Algorithm::DotProduct,
                AlgorithmHeapType::MAX(MaxHeap::new(heap_capacity)),
            ),
            "euclidean_distance" => (
                Algorithm::Euclidean,
                AlgorithmHeapType::MIN(MinHeap::new(heap_capacity)),
            ),
            _other => panic!("Not a valid Algorithm choice"),
        }
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
    fn test_teststore_find_top_3_similar_words_using_cosine_similarity() {
        let sentences_vectors = word_to_vector();

        let first_vector =
            ndarray::Array1::<f64>::from_vec(sentences_vectors.get(SEACH_TEXT).unwrap().to_owned());

        let mut search_list = vec![];

        for sentence in SENTENCES.iter() {
            let second_vector = ndarray::Array1::<f64>::from_vec(
                sentences_vectors.get(*sentence).unwrap().to_owned(),
            );

            search_list.push(second_vector)
        }

        let no_similar_values: usize = 3;

        let test_store = TestStore::new();

        let similar_n_search = test_store.find_similar_n(
            &first_vector,
            search_list.iter(),
            "cosine_similarity",
            no_similar_values,
        );

        let similar_n_vecs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>> =
            similar_n_search
                .into_iter()
                .map(|(vector, _)| vector)
                .collect();

        let most_similar_sentences_vec: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>> =
            MOST_SIMILAR
                .iter()
                .map(|sentence| {
                    ndarray::Array1::<f64>::from_vec(
                        sentences_vectors.get(*sentence).unwrap().to_owned(),
                    )
                })
                .collect();
        assert_eq!(most_similar_sentences_vec, similar_n_vecs)
    }
}
