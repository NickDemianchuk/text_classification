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
        self.f_counter = self.drop_ignored(
                self.split_docs(docs)
            )

    def transform_list(self, list):
        f_dict = {}
        for i, sub_list in enumerate(list):
            f_dict[sub_list[0]] = i
        return f_dict

    def drop_ignored(self, dict):
        for word in self.ignored_words:
            if word in dict:
                del dict[word]
        return dict

    def vectorize(self, docs, f_num):
        # Number of samples
        s_num = len(docs)
        # Creating a numpy array of shape [num of samples x num of features]
        v_arr = np.zeros(shape=[s_num, f_num])
        # Getting a list of lists of words and their occurrences
        f_list = self.f_counter.most_common(f_num)
        # Transforming the list into a dictionary of words with corresponding indexes
        # for later use in filling out the numpy array
        f_dict = self.transform_list(f_list)

        for i, doc in enumerate(docs):
            # Tokenizing each document in docs
            w_counter = self.split_doc(doc)
            # Filling out the numpy array
            for key, val in w_counter.items():
                if key in f_dict:
                    v_arr[i][f_dict[key]] = val
        return v_arr

    def split_docs(self, docs):
        c = Counter()
        for doc in docs:
            c += self.split_doc(doc)
        return c

    def split_doc(self, doc):
        return Counter(re.findall('\w+', doc))
