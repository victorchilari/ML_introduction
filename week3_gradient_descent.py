import pandas
import numpy as np
from math import exp
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

K = 0.1
ERROR = 1e-5


def sigma_y(i, w1, w2):
    return 1. / (1. + exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i])))


def delta_for_w(w_index, w1, w2, C):
    addition = sum((
        y[i] * X[w_index][i] * (1. - sigma_y(i, w1, w2)) for i in np.arange(0, len(y))
    ))
    addition *= K / len(y)
    addition -= K * C * (w1 if w_index == 1 else w2)

    return addition


def gradient_regression(C, iterations_remaining=10000):
    changed_w1, changed_w2 = 0., 0.
    while iterations_remaining:
        iterations_remaining -= 1
        w1, w2 = changed_w1, changed_w2
        changed_w1 = w1 + delta_for_w(1, w1, w2, C)
        changed_w2 = w2 + delta_for_w(2, w1, w2, C)
        if np.sqrt(mean_squared_error([w1, w2], [changed_w1, changed_w2])) <= ERROR:
            break
    return changed_w1, changed_w2


def sigma(xi, w1, w2):
    return 1. / (1 + np.exp(-w1 * xi[1] - w2 * xi[2]))


# %%
data = pandas.read_csv('data-logistic.csv', header=None)
X = data.loc[:, 1:]
y = data[0]

w1, w2 = gradient_regression(0.)
l2w1, l2w2 = gradient_regression(10.)

print(w1, w2, l2w1, l2w2)
# %%

scores = X.apply(lambda xi: sigma(xi, w1, w2), axis=1)
l2scores = X.apply(lambda xi: sigma(xi, l2w1, l2w2), axis=1)

auc_score = roc_auc_score(y, scores)
l2_auc_score = roc_auc_score(y, l2scores)

print(auc_score)
print(l2_auc_score)

# %%

f = open('Src 3.txt', 'w')
f.write(str(auc_score) + ' ' + str(l2_auc_score))
f.close()
