import numpy as np
# 梯度下降

class WeightedStackedEnsemble:
    def __init__(self, lr=0.01, num_iter=1000,tol=1e-5):
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.w = None




    def fit(self, S, y):
        n_samples, d = S.shape
        self.w = np.zeros(d)

        for _ in range(self.num_iter):
            y_pred = S.dot(self.w)
            grad = -(1.0 / n_samples) * S.T.dot(y - y_pred)
            self.w -= self.lr * grad

    def predict(self, S):
        return S.dot(self.w)