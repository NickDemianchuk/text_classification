import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def validate(data_set, folds, c_pow, penalty):
    fold_size = int(data_set.shape[0] / folds)
    np.random.shuffle(data_set)

    acc_sum = 0

    for i in range(folds):
        test = data_set[(fold_size * i):(fold_size * (i + 1)), :]
        pre = data_set[:(fold_size * i), :]
        post = data_set[(fold_size * (i + 1)):, :]
        train = np.concatenate((pre, post), axis=0)

        x_train, y_train = train[:, :-1], train[:, -1:]
        x_test, y_test = test[:, :-1], test[:, -1:]

        lr = LogisticRegression(C=10 ** c_pow, penalty=penalty, solver='liblinear')
        lr.fit(x_train, y_train.ravel())

        acc_sum += accuracy_score(y_test.ravel(), lr.predict(x_test))

    return acc_sum / folds
