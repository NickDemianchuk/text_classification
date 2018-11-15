import numpy as np

import k_fold_validator as kv


def grid_search(X, y, f_num, v):
    X_vct = v.vectorize(docs=X, f_num=f_num * 1000)

    # Concatenating labels to not 'lose' during shuffling
    X_vct = np.concatenate((X_vct, y.reshape(y.shape[0], 1)), axis=1)

    best_acc = 0
    for c_pow in range(-3, 3):
        for penalty in ['l1', 'l2']:
            # Average accuracy after 10-fold cross validation
            acc = kv.validate(data_set=X_vct, folds=10, c_pow=c_pow, penalty=penalty)
            if acc > best_acc:
                best_acc = acc
                best_c = 10 ** c_pow
                best_penalty = penalty
                best_f_num = f_num * 1000

    return [best_acc, best_c, best_penalty, best_f_num]
