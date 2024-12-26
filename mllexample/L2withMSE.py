import numpy as np


class WeightedStackedEnsemble:
    def __init__(self, lr=0.01, num_iter=1000, tol=1e-4):
        """
        lr: 学习率
        num_iter: 迭代次数
        tol: 判定“损失改善”是否足够大的阈值
        """
        self.lr = lr
        self.num_iter = num_iter
        self.tol = tol
        self.w = None

    def _mse(self, S, y, w):
        """ 计算 MSE 损失: (1/n)* sum( (y - S w)^2 ) """
        n_samples = len(y)
        y_pred = S.dot(w)
        return np.mean((y - y_pred) ** 2)

    def fit(self, S, y):
        n_samples, d = S.shape
        self.w = np.zeros(d)

        # 计算初始损失
        best_loss = self._mse(S, y, self.w)

        for _ in range(self.num_iter):
            old_w = self.w.copy()
            old_loss = best_loss

            # 计算梯度
            y_pred = S.dot(self.w)
            grad = -(1.0 / n_samples) * S.T.dot(y - y_pred)

            # 先试着更新
            self.w -= self.lr * grad

            # 计算新损失
            new_loss = self._mse(S, y, self.w)

            # 如果损失没降低，或者降低幅度太小 -> 回退
            if (old_loss - new_loss) < self.tol:
                self.w = old_w
                # 可以选择 break 或者继续迭代，这里直接 break 停止训练
                break
            else:
                # 否则接受新 w，并更新 best_loss
                best_loss = new_loss

    def predict(self, S):
        return S.dot(self.w)
