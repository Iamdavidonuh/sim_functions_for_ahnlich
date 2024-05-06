# sim_functions_for_ahnlich

A proof of concept on similarity functions for ahnlich.

This repo tries to implement various algorithms to be used by ahnlich to check for vector similarities.


Below are the algorithms

- Euclidean similarity
- dot-product

- Cosine similarity, etc


# Dot Product

The dot product or scalar product is an algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors), and returns a single number.

An Implementation for most similar items would be a MaxHeap, The larger the dot product between two vectors, the more similar



# COSINE SIMILARITY
Cosine similiarity is the cosine of the angles between vectors.
It tries to find how close or similar two vector points are.
It is scalent invarient meaning it is unaffected by the length of the vectors.


Cosine of the angles between vectors shows how similar, dissimilar or orthogonal(independent).


The range of similarity for cosine similarity ranges from -1 to 1:
where:
    1 means similar
    -1 different.
    0 means Orthogonal


To calculate the cosine similarity for two vectors, we need to:
- Calculate the dot product of both vectors
- Find the product of the magnitude of both vectors.
   A magnitude of a vector can be calculated using pythagoras theorem
   which is sqrt( A^2 + B^2) where A and B are two vectors.

- divide the dot product by the product of the magnitude of both vectors.
    ```
        cos(0) = A.B / ||A||.||B||
    ```

An Implementation for most similar items would be a MinHeap, The smaller the distance between two points, denotes higher similarity

# Euclidean similarity


 d(p,q)= sqrt { (p-q)^2 }
Euclidean distance is the square root of the sum of squared differences between corresponding
elements of the two vectors.

Note that the formula treats the values of X and Y seriously: no adjustment is made for
differences in scale. Euclidean distance is only appropriate for data measured on the same
scale(meaning it is  scale varient).
The distance derived is purely based on the difference between both vectors and is therefore,
prone to skewness if the units of the vectors are vastly different.

Hence, it is important to ensure that the data is normalised before applying the Euclidean distance function.
Additionally due to the curse of dimensionality, the effectiveness of Euclidean distance breaks down as 
dimensionality increases.



An Implementation for most similar items would be a MinHeap, The smaller the distance between two points, denotes higher similarity


