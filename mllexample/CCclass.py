import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from L2 import WeightedStackedEnsemble
from u_mReadData import *

class CCclass:
    def __init__(self):
        self.classifiers = []
        self.CC_cfl_test = []
        self.all_clf_train = []
        self.all_clf_test = []


    def train(self, X, Y):
        L = Y.shape[1]
        N = X.shape[0]

        for j in range(L):
            # D'j
            D_j_prime_x = []
            D_j_prime_y = []
            # for (x, y) ∈ D
            for i in range(N):
                #do x' ← [x1,...,xd ,y1,...,yj−1]
                if j == 0:
                    x_prime = X[i]
                else:
                    x_prime = np.concatenate((X[i], Y[i,:j]))
                # Dj' ← Dj ∪ (x' ,yj )
                D_j_prime_x.append(x_prime)
                D_j_prime_y.append(Y[i,j])

            D_j_prime_x = np.array(D_j_prime_x)
            D_j_prime_y = np.array(D_j_prime_y)
            # train hj to predict binary relevance of yj
            # P(y=1∣X)= 1/ (1+a) a = e的-（wX+b）次方
            clf = LogisticRegression()
            clf.fit(D_j_prime_x, D_j_prime_y)
            self.classifiers.append(clf)
            self.all_clf_train.append(clf)


    def CC_test(self, X, Y):

        LT = Y.shape[1]
        NT = X.shape[0]
        y_hat = []
        # for j = 1,...,L
        for j in range(LT):
            D_j_prime_x = []
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            for i in range(NT):
                if j == 0:
                    x_prime = X[i]
                else:
                    x_prime = np.concatenate((X[i], y_hat))
                D_j_prime_x.append(x_prime)

            D_j_prime_x = np.array(D_j_prime_x)

            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            y_pred_j = self.classifiers[j].predict(D_j_prime_x)[0]

            # return y
            self.CC_cfl_test.append(y_pred_j)
            return np.array(self.CC_cfl_test).T
# 训练元学习器
    def CC_train_BR_train(self, X, Y):
        for i in range(Y.shape[1]):
            print(Y.shape[1])
            #        cfl = WeightedStackedEnsemble()
            cfl = WeightedStackedEnsemble()
            cfl.fit(X, Y[:, i])
            self.all_clf_test.append(cfl)
# 测试基学习器
    def test_BRC_test(self, X, star, end):
        y_hat = []
        for i in range(star, end):
            D_j_prime_x = []
            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            for j in range(X):
                if j == 0:
                    x_prime = X[j]
                else:
                    x_prime = np.concatenate((X[j], y_hat))
                D_j_prime_x.append(x_prime)

            D_j_prime_x = np.array(D_j_prime_x)

            # do x' ← [x1,...,xd , yˆ1,..., yˆj−1]
            y_pred_j = self.classifiers[i].predict(D_j_prime_x)[0]

            # return y
            y_hat.append(y_pred_j)
            y_hat = np.array(y_hat).flatten()
            y_hat = y_hat.tolist()


    def CCCLF_clear(self):
        self.classifiers = []

    def BRCLF_test_clear(self):
        self.CC_cfl_test = []




