import os
import re
from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import List

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stopwords = stopwords.words('english')
stemmer = PorterStemmer()

DIR_NAME = 'data'
DICTIONARY_FILENAME = 'dictionary.txt'
OUTPUT_DIR_NAME = 'output'


class Preprocessor:
    def __init__(self, documents, stopwords, stemmer):
        self.stopwords = stopwords
        self.stemmer = stemmer
        if isinstance(documents, list):
            self.documents = documents
        else:
            self.documents = [documents]

    def preprocess(self):
        result = []
        for document in self.documents:
            tokens = self.tokenization(document)
            lower_tokens = self.lower(tokens)
            stemmed_tokens = self.stem(lower_tokens)
            rm_stopword_tokens = self.stopword_removal(stemmed_tokens)
            result.append(rm_stopword_tokens)
        return result

    def tokenization(self, document: str):
        # return [item for item in re.split(r' |\r\n|\n|,|\.|\'|,|"|\$|\+|\(|\)|!|%|&|`|-|\*|;|\?|/|:|_|@|#', document) if item]
        return [item for item in re.split(r'[ \r\n,./\'"~!@#$%^&*()_`1234567890:;{}?\[\]+-]', document) if item]

    def lower(self, tokens):
        return [token.lower() for token in tokens]

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def stopword_removal(self, words: List[str]):
        return [word for word in words if word not in self.stopwords]


@dataclass
class Term:
    index: int
    term: str
    frequency: int


class TFIDFVectorizer:
    def __init__(self, documents: List):
        self.documents: List = documents

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def tf(self):
        index_dict = self.get_index_dict()
        tf_vectors = []
        for document in self.documents:
            tf_vector = np.zeros(len(index_dict))
            for token in document:
                tf_vector[index_dict[token]] += 1
            tf_vectors.append(tf_vector)
        return tf_vectors

    @property
    def idf(self):
        terms = self.get_terms()
        idf_vector = np.zeros(len(terms))
        for term in terms:
            idf_vector[term.index] = log(
                self.document_count / term.frequency, 10)
        return idf_vector

    @property
    def tfidf(self):
        tf_vectors = self.tf
        idf_vector = self.idf
        tf_idf_vectors = [tf_vector * idf_vector for tf_vector in tf_vectors]
        tf_idf_vectors = [
            tf_idf_vector / sum(tf_idf_vector ** 2) ** 0.5 for tf_idf_vector in tf_idf_vectors]
        return tf_idf_vectors

    def get_index_dict(self) -> dict:
        terms = self.get_terms()
        return {term.term: term.index for term in terms}

    def get_terms(self, filename=None) -> List[Term]:
        dictionary = defaultdict(int)
        for document in self.documents:
            for token in set(document):
                dictionary[token] += 1
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[0])
        terms = [Term(index=i, term=term, frequency=frequency)
                 for i, (term, frequency) in enumerate(sorted_dict)]
        if filename:
            with open(DICTIONARY_FILENAME, 'w') as file:
                file.write('t_index\tterm\tdf\n')
                for term in terms:
                    file.write(
                        f'{term.index+1}\t{term.term}\t{term.frequency}\n')
        return terms

    TF = tf
    IDF = idf
    TFIDF = tfidf


if __name__ == '__main__':
    # 1. Load documents
    filenames = os.listdir(DIR_NAME)
    filenames = [filename for filename in filenames if filename[-4:] == '.txt']
    filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))
    documents = []
    for filename in filenames:
        with open(f'{DIR_NAME}/{filename}', 'r') as file:
            documents.append(file.read())

    # 2. Preprocess documents
    results = Preprocessor(documents=documents,
                           stopwords=stopwords, stemmer=stemmer).preprocess()

    # 3. Construct dictionary
    tfidf_vectorizer = TFIDFVectorizer(documents=results)
    dictionary = tfidf_vectorizer.get_terms(filename=DICTIONARY_FILENAME)

    # 4. Calculate tf-idf vectors
    tfidf_vecs = tfidf_vectorizer.tfidf

    # 5. Save tf-idf vectors
    if not os.path.exists(OUTPUT_DIR_NAME):
        os.mkdir(OUTPUT_DIR_NAME)

    for i, tfidf_vec in enumerate(tfidf_vecs):
        with open(f"{OUTPUT_DIR_NAME}/doc{i + 1}.txt", 'w') as file:
            term_count = sum([tfidf > 0 for tfidf in tfidf_vec])
            file.write(f"{term_count}\n")
            file.write("t_index\ttf-idf\n")
            for t_index, tfidf in enumerate(tfidf_vec):
                if tfidf == 0:
                    continue
                file.write(f"{t_index + 1}\t{tfidf}\n")

    # 6. Define cosine similarity function
    def next_line(doc_instance):
        try:
            index, val = map(float, doc_instance.readline().split('\t'))
        except ValueError:
            index, val = 0, 0
        return index, val

    def cosine_similarity(doc_x_path: str, doc_y_path: str) -> float:
        with open(doc_x_path, 'r') as doc_x, \
                open(doc_y_path, 'r') as doc_y:
            # skip first two rows
            next(doc_x), next(doc_y)
            next(doc_x), next(doc_y)

            x_index, x_val = next_line(doc_x)
            y_index, y_val = next_line(doc_y)
            num = 0
            denom_x, denom_y = x_val ** 2, y_val ** 2
            while True:
                if not x_index and not y_index:
                    break
                if x_index == y_index:
                    num += x_val * y_val
                    x_index, x_val = next_line(doc_x)
                    y_index, y_val = next_line(doc_y)
                    denom_x, denom_y = denom_x + x_val ** 2, denom_y + y_val ** 2
                elif not y_index or (x_index and (x_index < y_index)):
                    x_index, x_val = next_line(doc_x)
                    denom_x += x_val ** 2
                else:
                    y_index, y_val = next_line(doc_y)
                    denom_y += y_val ** 2
            return num / (denom_x ** 0.5 * denom_y ** 0.5)

    # 7. Calculate cosine similarity of doc1 and doc2
    similarity = cosine_similarity(
        f'{OUTPUT_DIR_NAME}/doc1.txt', f'{OUTPUT_DIR_NAME}/doc2.txt')
    print(f"{similarity = }")
