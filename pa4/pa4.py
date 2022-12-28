import os
import re
import time
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
        result = []
        for document in self.documents:
            tokens = self.tokenization(document)
            lower_tokens = self.lower(tokens)
            stemmed_tokens = self.stem(lower_tokens)
            rm_stopword_tokens = self.stopword_removal(stemmed_tokens)
            result.append(rm_stopword_tokens)
        return result

    @staticmethod
    def tokenization(document: str):
        # return [item for item in re.split(r'[ \r\n,./\'"~!@#$%^&*()_`1234567890:;{}?\[\]+-]', document) if item]
        doc = document.replace("\s+", " ").replace("\n",
                                                   "").replace("\r\n", "")
        doc = re.sub(r"[^\w\s]", "", doc)
        doc = doc.split(" ")
        return doc

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
        tf_vectors = []
        for document in self.documents:
            tf_vector = np.zeros(len(index_dict))
            for token in document:
                tf_vector[index_dict[token]] += 1
            tf_vectors.append(tf_vector)
        self._tf = tf_vectors
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
    def IDF_dict(self):
        terms = self.get_terms()
        idf_vector = {}
        for term in terms:
            idf_vector[term.term] = log(
                self.document_count / term.frequency, 10)
        return idf_vector

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
    def TFIDF_dict(self):
        tf_vectors = self.TF_dict
        idf_vector = self.IDF_dict
        tf_idf_vectors = [tf_vector * idf_vector for tf_vector in tf_vectors]
        tf_idf_vectors = [
            tf_idf_vector / sum(tf_idf_vector ** 2) ** 0.5 for tf_idf_vector in tf_idf_vectors]
        self.tfidf = tf_idf_vectors
        return tf_idf_vectors

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
        for document in self.documents:
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


def cosine(tfidf1, tfidf2):
    return np.dot(tfidf1, tfidf2)


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
        docs_count = len(self.training_ids)

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
                    [log(self.cond_prob[term.term][category], 10) * tf[doc_id][term.term] for term in vocabulary if term.term in tf[doc_id]])
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
                    for i in range(0, 2):
                        for j in range(0, 2):
                            n[i][j] = sum([(term.term in tf[doc_id]) == j and (
                                doc_id in self.category_to_doc_ids[category]) == i for doc_id in self.training_ids])
                    N = n.sum()
                    for i in range(2):
                        for j in range(2):
                            E = n.sum(axis=0)[j] * n.sum(axis=1)[i] / N
                            term_chi += (n[i][j] - E) ** 2 / N
                chi[term] = term_chi
            vocabulary = sorted(chi, key=chi.get, reverse=True)[:500]
            return vocabulary
        else:
            raise Exception('Feature selection type not implemented.')


@dataclass
class Item:
    index: int
    similarity: float

    def __gt__(self, x):
        if self.similarity > x.similarity:
            return True
        elif self.similarity == x.similarity:
            return self.index > x.index
        return False

    def __mul__(self, x):
        return Item(self.index, x * self.similarity)

    __rmul__ = __mul__


class Queue:
    def __init__(self, q=None):
        self.queue = q or []

    def is_empty(self):
        return len(self.queue) == 0

    @property
    def front(self):
        return self.queue[0]

    def push(self, x):
        self.queue.append(x)

    def pop(self):
        item, *self.queue = self.queue
        return item

    def delete(self, x):
        self.queue.remove(x)


class PriorityQueue(Queue):
    def __init__(self, q=None):
        super().__init__(q)
        self.queue = sorted(self.queue, reverse=True)

    @property
    def key(self):
        return lambda x: -1 * x.similarity

    def push(self, x):
        pos = self._get_insert_position(self.key(x), key=self.key)
        self.queue.insert(pos, x)

    def _get_insert_position(self, x, key=None):
        q = self.queue
        l, r = 0, len(q)
        while l < r:
            mid = (l + r) // 2
            if x < key(q[mid]):
                r = mid
            else:
                l = mid + 1
        return l

    def _get_element_position(self, x):
        q = self.queue
        l, r = 0, len(self.queue)
        while l <= r:
            mid = (l + r) // 2
            if q[mid] > x:
                l = mid + 1
            elif q[mid] < x:
                r = mid - 1
            elif q[mid] == x:
                return mid
            elif q[l] == x:
                return l
            elif q[r] == x:
                return r
        raise Exception('not found')

    def delete(self, x):
        self.queue.remove(x)


class HAC:
    def __init__(self, tfidfs):
        self.tfidfs = tfidfs
        doc_count = len(tfidfs)
        self.doc_count = doc_count
        self.C = []
        self.I = np.ones(doc_count)
        self.P = []
        self.A = []
        self.init()

    def init(self):
        for n, tfidf_n in enumerate(self.tfidfs):
            self.C.append([])
            for i, tfidf_i in enumerate(self.tfidfs):
                self.C[n].append(Item(i, cosine(tfidf_n, tfidf_i)))
            self.P.append(PriorityQueue(self.C[n]))
            self.P[n].delete(self.C[n][n])

    def get_most_similar_documents(self):
        max_sim = -1
        document_pair = None
        for i, p in enumerate(self.P):
            if not self.I[i]:
                continue
            if p.front.similarity > max_sim:
                item = p.front
                max_sim = item.similarity
                document_pair = i, item.index
        return document_pair

    def classify(self):
        start = time.time()
        for k in range(self.doc_count - 1):
            k1, k2 = self.get_most_similar_documents()
            self.A.append((k1, k2))
            self.I[k2] = 0
            self.P[k1] = PriorityQueue()
            for i in range(len(self.I)):
                if self.I[i] == 1 and i != k1:
                    self.P[i].delete(self.C[i][k1])
                    self.P[i].delete(self.C[i][k2])
                    self.C[i][k1].similarity = min(
                        self.C[i][k1].similarity, self.C[i][k2].similarity)
                    self.P[i].push(self.C[i][k1])
                    self.C[k1][i].similarity = min(
                        self.C[i][k1].similarity, self.C[i][k2].similarity)
                    self.P[k1].push(self.C[k1][i])
        print(f"done after {time.time() - start}s")

    def save_result(self, cluster_counts):
        A = self.A
        merge_result = {i: [i] for i in range(len(A) + 1)}
        for i, (c1, c2) in enumerate(A):
            merge_result[c1].extend(merge_result[c2].copy())
            merge_result.pop(c2)
            # print(i, len(merge_result), len(merge_result[0]))
            if len(merge_result) in cluster_counts:
                with open(f"{len(merge_result)}.txt", "w") as file:
                    for cluster in merge_result.values():
                        for doc_id in sorted(cluster):
                            file.write(f"{doc_id + 1}\n")
                        file.write('\n')


DIR_NAME = './data'

if __name__ == '__main__':

    filenames = sorted([filename for filename in os.listdir(DIR_NAME) if re.match(
        '\d+.txt', filename)], key=lambda x: int(re.findall('\d+', x)[0]))
    documents = []
    for filename in filenames:
        with open(f'{DIR_NAME}/{filename}', 'r') as file:
            documents.append(file.read())

    preprocessed_data = Preprocessor(
        documents=documents, stopwords=stopwords, stemmer=stemmer).preprocess()
    tfidf_vec = TFIDFVectorizer(preprocessed_data)
    tfidf = tfidf_vec.TFIDF

    hac_model = HAC(tfidf)
    hac_model.classify()
    hac_model.save_result([8, 13, 20])
