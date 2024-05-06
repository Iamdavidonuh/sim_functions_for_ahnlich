use ndarray::prelude::*;
mod generator;

fn cosine_similarity(
    first: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
    second: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
) -> f64 {
    // formular = dot product of vectors / product of the magnitude of the vectors
    // maginiture of a vector can be calcuated using pythagoras theorem.
    // sqrt of sum of vector values
    // for magnitutude of A( ||A||) =[1,2]
    // sqrt[(1)^2 + (2)^2]
    //
    //

    let dot_product = dot_product(&first, &second);

    // the magnitude can be calculated using the arr.norm method.
    //

    let mag_first = &first
        .iter()
        .map(|x| x * x)
        .map(|x| x as f64)
        .sum::<f64>()
        .sqrt();

    let mag_second = &second
        .iter()
        .map(|x| x * x)
        .map(|x| x as f64)
        .sum::<f64>()
        .sqrt();

    let cosine_sim = dot_product / (mag_first * mag_second);

    cosine_sim
}
// for mulitplication of a matrix, the number of columns in the first matrix == the number of rows
// in the second
// A.B != B.A

fn dot_product(
    first: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
    second: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
) -> f64 {
    // transpose if the shapes are equal that is
    // check that for a A = m*n and B = p * q
    // m = q and n = q
    // else check that it's already transposed

    if first.shape() != second.shape() {
        panic!("Incorrect dimensions, both vectors must match");
    }

    let dot_product = second.dot(&first.t());

    dot_product
}

fn euclidean_distance(
    first: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
    second: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, Ix1>,
) -> f64 {
    // Calculate the sum of squared differences for each dimension
    let mut sum_of_squared_differences = 0.0;
    for (&coord1, &coord2) in first.iter().zip(second.iter()) {
        let diff = coord1 - coord2;
        sum_of_squared_differences += diff * diff;
    }

    // Calculate the square root of the sum of squared differences
    let distance = f64::sqrt(sum_of_squared_differences);
    distance
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEACH_TEXT: &'static str =
        "Football fans enjoy gathering to watch matches at sports bars.";

    const MOST_SIMILAR: [&'static str; 3] = [
        "Attending football games at the stadium is an exciting experience.",
        "On sunny days, people often gather outdoors for a friendly game of football.",
        "Rainy weather can sometimes lead to canceled outdoor events like football matches.",
    ];
    const SENTENCES: [&'static str; 5] = [
        "On sunny days, people often gather outdoors for a friendly game of football.",
        "Attending football games at the stadium is an exciting experience.",
        "Grilling burgers and hot dogs is a popular activity during summer barbecues.",
        "Rainy weather can sometimes lead to canceled outdoor events like football matches.",
        "Sunny weather is ideal for outdoor activities like playing football.",
    ];

    #[test]
    fn test_cosine_similarity_on_same_vectors() {
        let first_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);
        let second_vector = ndarray::Array1::<f64>::zeros(2).map(|x| x + 2.0);

        let similarity = cosine_similarity(&first_vector, &second_vector).round();

        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_find_top_3_similar_words_using_cosine_similarity() {
        let sentences_vectors = generator::word_to_vector();

        let mut most_similar_result: Vec<(&'static str, f64)> = vec![];

        let first_vector =
            ndarray::Array1::<f64>::from_vec(sentences_vectors.get(SEACH_TEXT).unwrap().to_owned());

        for sentence in SENTENCES.iter() {
            let second_vector = ndarray::Array1::<f64>::from_vec(
                sentences_vectors.get(*sentence).unwrap().to_owned(),
            );

            let similarity = cosine_similarity(&first_vector, &second_vector);

            most_similar_result.push((*sentence, similarity))
        }

        // sort by largest first, the closer to zero the more similar it is
        most_similar_result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let final_result: Vec<&'static str> = most_similar_result.iter().map(|s| s.0).collect();

        assert_eq!(MOST_SIMILAR.to_vec(), final_result[..3]);
    }
    #[test]
    fn test_find_top_3_similar_words_using_euclidean_distance() {
        let sentences_vectors = generator::word_to_vector();

        let mut most_similar_result: Vec<(&'static str, f64)> = vec![];

        let first_vector =
            ndarray::Array1::<f64>::from_vec(sentences_vectors.get(SEACH_TEXT).unwrap().to_owned());

        for sentence in SENTENCES.iter() {
            let second_vector = ndarray::Array1::<f64>::from_vec(
                sentences_vectors.get(*sentence).unwrap().to_owned(),
            );

            let similarity = euclidean_distance(&first_vector, &second_vector);

            most_similar_result.push((*sentence, similarity))
        }

        // sort by smallest first, the closer to zero the more similar it is
        most_similar_result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let final_result: Vec<&'static str> = most_similar_result.iter().map(|s| s.0).collect();

        assert_eq!(MOST_SIMILAR.to_vec(), final_result[..3]);
    }

    #[test]
    fn test_find_top_3_similar_words_using_dot_product() {
        let sentences_vectors = generator::word_to_vector();

        let mut most_similar_result: Vec<(&'static str, f64)> = vec![];

        let first_vector =
            ndarray::Array1::<f64>::from_vec(sentences_vectors.get(SEACH_TEXT).unwrap().to_owned());

        for sentence in SENTENCES.iter() {
            let second_vector = ndarray::Array1::<f64>::from_vec(
                sentences_vectors.get(*sentence).unwrap().to_owned(),
            );

            let similarity = dot_product(&first_vector, &second_vector);

            most_similar_result.push((*sentence, similarity))
        }

        // sort by largest first, the closer to zero the more similar it is
        most_similar_result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let final_result: Vec<&'static str> = most_similar_result.iter().map(|s| s.0).collect();

        assert_eq!(MOST_SIMILAR.to_vec(), final_result[..3]);
    }
}
