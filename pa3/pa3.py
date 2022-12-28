import csv
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


class Preprocessor:
    def __init__(self, documents, stopwords, stemmer):
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.documents = documents

    def preprocess(self):
        result = {}
        for doc_id, document in self.documents.items():
            tokens = self.tokenization(document)
            lower_tokens = self.lower(tokens)
            stemmed_tokens = self.stem(lower_tokens)
            rm_stopword_tokens = self.stopword_removal(stemmed_tokens)
            result[doc_id] = rm_stopword_tokens
        return result

    @staticmethod
    def tokenization(document: str):
        return [item for item in re.split(r'[ \r\n,./\'"~!@#$%^&*()_`1234567890:;{}?\[\]+-]', document) if item]

    @staticmethod
    def lower(tokens):
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

    def __hash__(self):
        return hash(self.index)


class TFIDFVectorizer:
    def __init__(self, documents):
        self.documents = documents
        self._tf = None
        self._idf = None
        self._tfidf = None

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def TF(self):
        if self._tf:
            return self._tf

        index_dict = self.get_index_dict()
        tf_vectors = {}
        for doc_id, document in self.documents.items():
            tf_vector = np.zeros(len(index_dict))
            for token in document:
                tf_vector[index_dict[token]] += 1
            tf_vectors[doc_id] = tf_vector
        self.tf = tf_vectors
        return tf_vectors

    @property
    def TF_dict(self):
        tf_vectors = {}
        for doc_id, document in self.documents.items():
            tf_vector = {}
            for token in document:
                if token in tf_vector:
                    tf_vector[token] += 1
                else:
                    tf_vector[token] = 1
            tf_vectors[doc_id] = tf_vector
        return tf_vectors

    @property
    def IDF(self):
        if self._idf:
            return self._idf

        terms = self.get_terms()
        idf_vector = np.zeros(len(terms))
        for term in terms:
            idf_vector[term.index] = log(
                self.document_count / term.frequency, 10)
        self.idf = idf_vector
        return idf_vector

    @property
    def TFIDF(self):
        if self._tfidf:
            return self._tfidf

        tf_vectors = self.TF
        idf_vector = self.IDF
        tf_idf_vectors = [tf_vector * idf_vector for tf_vector in tf_vectors]
        tf_idf_vectors = [
            tf_idf_vector / sum(tf_idf_vector ** 2) ** 0.5 for tf_idf_vector in tf_idf_vectors]
        self.tfidf = tf_idf_vectors
        return tf_idf_vectors

    def get_index_dict(self) -> dict:
        terms = self.get_terms()
        return {term.term: term.index for term in terms}

    def get_terms(self, filename=None) -> List[Term]:
        dictionary = defaultdict(int)
        for doc_id, document in self.documents.items():
            for token in set(document):
                dictionary[token] += 1
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[0])
        terms = [Term(index=i, term=term, frequency=frequency)
                 for i, (term, frequency) in enumerate(sorted_dict)]
        if filename:
            with open(filename, 'w') as file:
                file.write('t_index\tterm\tdf\n')
                for term in terms:
                    file.write(
                        f'{term.index+1}\t{term.term}\t{term.frequency}\n')
        return terms


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


class NBClassifier:
    def __init__(self, preprocessed_data, training_ids, testing_ids, training_data, selection_type='chi'):
        self.training_ids = training_ids
        self.testing_ids = testing_ids
        self.preprocessed_data = preprocessed_data
        training_docs = {
            doc_id: preprocessed_data[doc_id] for doc_id in training_ids}
        self.train_tfidf_vectorizer = TFIDFVectorizer(documents=training_docs)
        self.tf_dict = TFIDFVectorizer(
            documents=self.preprocessed_data).TF_dict
        self.category_to_doc_ids = training_data
        self.selection_type = selection_type
        self.feature = self.feature_selection()
        self.cond_prob = {term.term: dict() for term in self.feature}
        self.prior = {}

    @property
    def categories(self):
        return list(self.category_to_doc_ids.keys())

    def train(self):
        vocabulary = self.feature
        docs_count = len(training_ids)

        for category in self.categories:
            nc = len(self.category_to_doc_ids[category])
            self.prior[category] = nc / docs_count
            tct = {}
            tf = self.tf_dict
            for term in vocabulary:
                tokens_sum = 0
                doc_ids = self.category_to_doc_ids[category]
                tct[term.term] = sum([tf[doc_id][term.term]
                                     for doc_id in doc_ids if term.term in tf[doc_id]])

            for term in vocabulary:
                self.cond_prob[term.term][category] = (
                    tct[term.term] + 1) / (sum(tct.values()) + len(vocabulary))
        return self

    def pred(self):
        doc_category = {}
        vocabulary = self.feature
        for doc_id in self.testing_ids:
            score = {}
            tf = self.tf_dict
            for category in self.categories:
                score[category] = self.prior[category] + sum(
                    [log(self.cond_prob[term.term][category]) * tf[doc_id][term.term] for term in vocabulary if term.term in tf[doc_id]])
            doc_category[doc_id] = max(score, key=score.get)
        return doc_category

    def feature_selection(self):
        if self.selection_type == 'chi':
            vocabulary = self.train_tfidf_vectorizer.get_terms()
            tf = self.tf_dict
            chi = {}
            for term in vocabulary:
                term_chi = 0
                for category in self.categories:
                    n = np.zeros((2, 2))
                    n[1][1] = sum([term.term in tf[doc_id] and doc_id in self.category_to_doc_ids[category]
                                  for doc_id in self.training_ids])
                    n[1][0] = sum([term.term not in tf[doc_id] and doc_id in self.category_to_doc_ids[category]
                                  for doc_id in self.training_ids])
                    n[0][1] = sum([term.term in tf[doc_id] and doc_id not in self.category_to_doc_ids[category]
                                  for doc_id in self.training_ids])
                    n[0][0] = sum([term.term not in tf[doc_id] and doc_id not in self.category_to_doc_ids[category]
                                  for doc_id in self.training_ids])
                    N = n.sum()
                    for i in range(2):
                        for j in range(2):
                            E = n.sum(axis=0)[j] * n.sum(axis=1)[i] / N
                            term_chi += (n[i][j] - E) ** 2 / N
                chi[term] = term_chi
            vocabulary = sorted(chi, key=chi.get, reverse=True)[:500]
            return vocabulary
        else:
            raise Exception('Invalid feature selection type')


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Value'])
        for k, v in preds.items():
            # if p2 > p1:
            #     p1, p2 = p2, p1
            writer.writerow([k, v])


DIR_NAME = './data'
TRAINING_FILENAME = 'training.txt'
PRED_SAVE_PATH = 'pred.csv'

if __name__ == '__main__':
    # 1. Load documents
    training_data = {}
    training_ids = []
    with open(f"{DIR_NAME}/{TRAINING_FILENAME}", 'r') as file:
        for line in file.readlines():
            category, *doc_ids = line.split()
            doc_ids = list(map(int, doc_ids))
            training_ids.extend(doc_ids)
            training_data[int(category)] = doc_ids
    testing_ids = [doc_id for doc_id in range(
        1, 1096) if doc_id not in training_ids]
    filenames = sorted([filename for filename in os.listdir(DIR_NAME) if re.match(
        '\d+.txt', filename)], key=lambda x: int(re.findall('\d+', x)[0]))
    documents = {}
    for filename in filenames:
        with open(f'{DIR_NAME}/{filename}', 'r') as file:
            documents[int(re.findall('\d+', filename)[0])] = file.read()

    # 2. Preprocess data
    preprocessed_data = Preprocessor(
        documents=documents, stopwords=stopwords, stemmer=stemmer).preprocess()

    # 3. Init NBClassifier and train
    classifier = NBClassifier(preprocessed_data, training_ids,
                              testing_ids, training_data, selection_type='chi').train()

    # 4. Predict
    pred = classifier.pred()

    # 5. save prediction
    save_pred(pred, PRED_SAVE_PATH)
