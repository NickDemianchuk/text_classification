import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from document_reader import DocumentReader
from k_fold_validator import KFoldValidator
from vectorizer import Vectorizer

dr = DocumentReader()
df = dr.read(file_names=['data/rt-polarity.pos', 'data/rt-polarity.neg'],
             labels=[1, 0])
# Task 1
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:, 0], df.values[:, 1].astype('int'), test_size=0.2, random_state=0)

# Task 2
v = Vectorizer(docs=X_train)
kfv = KFoldValidator()

best_acc = 0
best_clf = None
for f_num in tqdm(range(1, 5)):
    X_train_v = v.vectorize(docs=X_train, f_num=f_num * 1000)
    X_train_v = np.concatenate((X_train_v, y_train.reshape(y_train.shape[0], 1)), axis=1)
    for pow in range(-4, 3):
        for penalty in ['l1', 'l2']:
            np.random.shuffle(X_train_v)
            lr = LogisticRegression(C=10 ** pow, solver='liblinear')
            acc = kfv.validate(data=X_train_v, k=10, clf=lr)
            if best_acc < acc:
                best_clf = lr
                best_acc = acc

X_test_v = v.vectorize(X_test, 5000)
best_clf.fit(X_test_v, y_test)
acc = accuracy_score(y_test, best_clf.predict(X_test_v))

print('The accuracy of the tuned LR evaluated on the test set is %.3f percent' % (best_acc * 100))
print('The parameters of the tuned LR:\n' + str(best_clf))
