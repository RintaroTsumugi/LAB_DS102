import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self._w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        # y, y_hat : (N, k)
        #return -(y * np.log(y_hat)).sum(axis = -1).mean()
        y_hat = np.clip(y_hat, 1e-15, 1.0 - 1e-15)
        return -(y * np.log(y_hat)).sum(axis=-1).mean()

    def fit(self, X: np.ndarray, y:np.ndarray):
        N,d = X.shape
        k = y.shape[1]

        self._w = np.zeros((d,k), dtype=np.float64)

        for _ in tqdm(range(self.epoch)):
            #forward class
            y_hat = self.predict(X)
            
            #backward class
            delta_y = (y_hat - y)
            gradient = (X.T @ delta_y) / N # (d,k)

            #update
            self._w = self._w - self.lr * gradient # (d, k)

            #loss
            loss_value = self.loss_fn(y, y_hat)
            self.losses.append(loss_value)

    def softmax(self, z: np.ndarray):
    # Trừ max để tránh lỗi số lớn (Numerical Stability)
        exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
        
        # Chia cho tổng theo hàng (axis=-1)
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    def evaluate(self, y, y_hat) -> dict:
        y_pred = np.argmax(y_hat, axis=1)
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def predict(self, X: np.ndarray):
        z = X @ self._w # (N, 1)
        y_hat = self.softmax(z)
        return y_hat