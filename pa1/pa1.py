import re
from typing import List

import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')


stopwords = stopwords.words('english')


stemmer = PorterStemmer()


def stopword_removal(words: List[str]):
    return [word for word in words if word not in stopwords]


def tokenization(document: str):
    return [item for item in re.split(' |\r\n|\n|,|\.|\'|,|"', document) if item]


SOURCE_URL = 'https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt'

if __name__ == '__main__':

    # 1. get document
    document = requests.get(SOURCE_URL).text

    # 2. tokenized document
    tokens = tokenization(document)

    # 3. lowercase tokens
    lower_token = [token.lower() for token in tokens]

    # 4. stemming tokens
    stemmed_token = [stemmer.stem(token) for token in tokens]

    # 5. stopwords removal
    result = stopword_removal(stemmed_token)

    # 6. export result to result.txt
    with open('result.txt', 'w') as file:
        file.write(' '.join(result))
