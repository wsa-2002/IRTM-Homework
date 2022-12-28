# IRTM Homework

### PA1

- Write a program to extract terms from a document.
  - Tokenization.
  - Lowercasing everything.
  - Stemming using Porter’s algorithm.
  - Stopword removal.
  - Save the result as a txt file.

### PA2

- Write a program to convert a set of documents into tf-idf vectors.
  - Construct a dictionary based on the terms extracted from the given documents.
  - Transfer each document into a tf-idf unit vector.
  - Write a function cosine($\text{Doc}_\text{x}$, $\text{Doc}_\text{y}$) which loads the tf-idf vectors of documents x and y and returns their cosine similarity.

### PA3

- Multinomial NB Classifier
  - Employ feature selection method and use only 500 terms for classification.
  - calculate P(X=t|c) by using add-one smoothing.

### PA4

- HAC clustering
  - Documents are represented as normalized tf-idf vectors.
  - Cosine similarity for pair-wise document similarity.
