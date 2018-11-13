import numpy as np
from sklearn.metrics import accuracy_score


class KFoldValidator(object):

    def validate(self, data_set, folds, clf):
        fold_size = int(data_set.shape[0] / folds)
        acc_sum = 0
        np.random.shuffle(data_set)
        for i in range(folds):
            test = data_set[(fold_size * i):(fold_size * (i + 1)), :]
            pre = data_set[:(fold_size * i), :]
            post = data_set[(fold_size * (i + 1)):, :]
            train = np.concatenate((pre, post), axis=0)
            x_train, y_train = train[:, :-1], train[:, -1:]
            x_test, y_test = test[:, :-1], test[:, -1:]
            clf.fit(x_train, y_train.ravel())
            acc_sum += accuracy_score(y_test.ravel(), clf.predict(x_test))
        return acc_sum / folds
