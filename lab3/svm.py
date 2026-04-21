import numpy as np
from tqdm import tqdm

class SVM :

    def __init__(self, C:float, epochs: int = 150, lr: float = 0.0001):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.b = None
        self.w = None
        self.losses = []
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape
        indices = np.arange(N)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        self.w = np.zeros(dim,) #(dim,1)
        self.b = 0
        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            for ith, x_i in enumerate(X):
                condition = y[ith] * (np.dot(x_i, self.w) + self.b)

                if condition >= 1:
                    # Only regularizatioin term
                    dw = self.w
                    db = 0
                else:
                    # Hinge loss active
                    dw = self.w - self.C * y[ith] * x_i
                    db = -self.C * y[ith]

                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            y_hat = self.predict(X)
            loss = self.hinge_loss(y, y_hat)
            self.losses.append(loss)

    def predict(self, X: np.ndarray):
        y_hat = self.w @ X.T + self.b
        return y_hat

    def hinge_loss(self, y: np.ndarray, y_hat: np.ndarray):
        delta = 1 - y * y_hat
        return 0.5 * (self.w.T @ self.w) + self.C * np.where(delta > 0, delta, 0).sum() # np.where hàm so sánh