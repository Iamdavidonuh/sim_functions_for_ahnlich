use ndarray::prelude::*;

use std::{collections::HashMap, f64};

use crate::types::{Algorithm, AlgorithmFunc, NonNanF64};

pub(super) fn word_to_vector() -> HashMap<String, ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>>
{
    let words = std::fs::read_to_string("test_json.json").unwrap();

    let words_to_vec: HashMap<String, Vec<f64>> = serde_json::from_str(&words).unwrap();

    HashMap::from_iter(
        words_to_vec
            .into_iter()
            .map(|(key, value)| (key, ndarray::Array1::<f64>::from_vec(value))),
    )
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

    fn key_to_words(
        key: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
        vector_to_sentences: &Vec<(ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>, String)>,
    ) -> Option<String> {
        for (vector, word) in vector_to_sentences {
            if key == vector {
                return Some(word.to_string());
            }
        }
        None
    }

    #[test]
    fn test_teststore_find_top_3_similar_words_using_find_nearest_n() {
        let sentences_vectors = word_to_vector();

        let vectors_to_sentences: Vec<_> = sentences_vectors
            .clone()
            .into_iter()
            .map(|(key, value)| (value, key))
            .collect();

        let first_vector = sentences_vectors.get(SEACH_TEXT).unwrap().to_owned();

        let mut search_list = vec![];

        for sentence in SENTENCES.iter() {
            let second_vector = sentences_vectors.get(*sentence).unwrap().to_owned();

            search_list.push(second_vector)
        }

        let no_similar_values: usize = 3;

        let test_store = TestStore::new();

        let similar_n_search = test_store.find_similar_n(
            &first_vector,
            search_list.iter(),
            &Algorithm::Cosine,
            no_similar_values,
        );

        let similar_n_vecs: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>> =
            similar_n_search
                .into_iter()
                .map(|(vector, _)| vector.to_owned())
                .collect();

        let most_similar_sentences_vec: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>> =
            MOST_SIMILAR
                .iter()
                .map(|sentence| sentences_vectors.get(*sentence).unwrap().to_owned())
                .collect();
        let cosine_sentences: Vec<_> = similar_n_vecs
            .iter()
            .map(|val| key_to_words(val, &vectors_to_sentences))
            .collect();

        println!("{cosine_sentences:?}");
        println!("{:?}", MOST_SIMILAR);

        assert_eq!(most_similar_sentences_vec, similar_n_vecs);
    }
}
