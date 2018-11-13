from statistics import mean

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score


class KFoldValidator(object):

    def validate(self, data, k, clf):
        fold_size = int(data.shape[0] / k)

        def processInput(i):
            test = data[fold_size * i:fold_size * (i + 1), :]
            pre = data[:fold_size * i, :]
            post = data[fold_size * (i + 1):, :]
            train = np.concatenate((pre, post), axis=0)
            x_train, y_train = train[:, :-1], train[:, -1:]
            x_test, y_test = test[:, :-1], test[:, -1:]
            clf.fit(x_train, y_train.ravel())
            return accuracy_score(y_test.ravel(), clf.predict(x_test))

        results = Parallel(n_jobs=4)(delayed(processInput)(i) for i in range(k))
        return mean(results)
