from operator import itemgetter

from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import data_reader as dr
import grid_search as gs
from vectorizer import Vectorizer

df = dr.read(file_names=['data/rt-polarity.pos', 'data/rt-polarity.neg'],
             labels=[1, 0])

# Task 1
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:, 0], df.values[:, 1].astype('int'), test_size=0.2, random_state=0)

# Task 2
v = Vectorizer(docs=X_train)

# Task 3-4
result = Parallel(n_jobs=4)(delayed(gs.grid_search)(X=X_train, y=y_train, f_num=f_num, v=v) for f_num in range(1, 6))
result = sorted(result, key=itemgetter(0), reverse=True)
best_c = result[0][1]
best_penalty = result[0][2]
best_f_num = result[0][3]

# Task 5
X_train_vct = v.vectorize(docs=X_train, f_num=best_f_num)
X_test_vct = v.vectorize(docs=X_test, f_num=best_f_num)

best_lr = LogisticRegression(C=best_c, penalty=best_penalty, solver='liblinear')
best_lr.fit(X_train_vct, y_train)

print('The accuracy is %.3f percent' % (accuracy_score(y_test, best_lr.predict(X_test_vct)) * 100))
print(best_lr)
