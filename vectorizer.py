import re
from collections import Counter

import numpy as np


class Vectorizer(object):
    ignored_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                     'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
                     'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from',
                     'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
                     'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                     'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
                     'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
                     'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                     'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
                     'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                     'further', 'was', 'here', 'than']

    def __init__(self, docs):
        self.token_counter = self.drop_ignored(
            self.split_docs(docs)
        )

    def transform_list(self, token_list):
        token_dict = {}
        for i, sub_list in enumerate(token_list):
            token_dict[sub_list[0]] = i
        return token_dict

    def drop_ignored(self, token_dict):
        for word in self.ignored_words:
            if word in token_dict:
                del token_dict[word]
        return token_dict

    def vectorize(self, docs, f_num):
        # A number of samples
        s_num = len(docs)
        # A numpy array of shape [num of samples x num of features]
        np_arr = np.zeros(shape=[s_num, f_num])
        # A list of lists [word, number of occurrences]
        token_list = self.token_counter.most_common(f_num)
        # A dictionary [word, index]
        token_dict = self.transform_list(token_list)

        for i, doc in enumerate(docs):
            # A Counter of tokens for each document
            token_counter = Counter(self.split_doc(doc))
            # Filling out the numpy array
            for key, val in token_counter.items():
                # Guard against stopwords
                if key in token_dict:
                    np_arr[i][token_dict[key]] = val
        return np_arr

    def split_docs(self, docs):
        tokens = []
        for doc in docs:
            tokens += self.split_doc(doc)
        return Counter(tokens)

    def split_doc(self, doc):
        return re.findall('\w+', doc)
