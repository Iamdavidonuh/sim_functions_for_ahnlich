use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize)]
struct VectorWord {
    key: String,
    value: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VectorWords {}

pub(super) fn word_to_vector() -> HashMap<String, Vec<f64>> {
    let words = std::fs::read_to_string("test_json.json").unwrap();

    let words_to_vec: HashMap<String, Vec<f64>> = serde_json::from_str(&words).unwrap();

    words_to_vec
}
