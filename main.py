import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from k_fold_validator import KFoldValidator
from vectorizer import Vectorizer


def read(file_names, labels):
    df = pd.DataFrame()
    for label, file_name in zip(labels, file_names):
        file = open(file_name, 'r')
        lines = file.readlines()
        for line in lines:
            df = df.append([[line, label]], ignore_index=True)
        file.close()
    return df


df = read(file_names=['data/rt-polarity.pos', 'data/rt-polarity.neg'],
          labels=[1, 0])

# Task 1
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:, 0], df.values[:, 1].astype('int'), test_size=0.2, random_state=0)

# Task 2
v = Vectorizer(docs=X_train)

# Task 3
kv = KFoldValidator()

# Task 4
best_acc = 0
for f_num in range(1, 6):
    # Vectorazing a dataset
    data_set = v.vectorize(docs=X_train, f_num=f_num * 1000)
    target = y_train
    # Concatenating labels to each sample for shuffling
    data_set = np.concatenate((data_set, target.reshape(target.shape[0], 1)), axis=1)
    for c_pow in range(-3, 3):
        lr = LogisticRegression(C=10 ** c_pow, solver='liblinear')
        acc = kv.validate(data_set=data_set, folds=10, clf=lr)
        if acc > best_acc:
            best_acc = acc
            best_clf = lr
            best_f_num = f_num * 1000
print('Best parameters for LR: ')
print(best_clf)


# Task 5
test = v.vectorize(X_test, best_f_num)
print('The accuracy is %.3f percent' % (accuracy_score(y_test, best_clf.predict(test)) * 100))
