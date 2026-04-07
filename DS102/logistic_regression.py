import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self._w = None
        self.losses = []

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        l = (1 - y) * np.log(1 - y_hat + 10e-15) + y * np.log(y_hat + 10e-15)
        return -l.mean()
    
    def fit(self, X: np.ndarray, y:np.ndarray):
        N, d = X.shape
        y = y.reshape(-1, 1)
        self._w = np.zeros((d,1), dtype=np.float64)

        for _ in tqdm(range(self.epoch)):
            #forward class
            y_hat = self.predict_prob(X)
            
            #backward class
            delta_y = y_hat - y
            gradient = (X.T @ delta_y) / N # (d,1)

            #update
            self._w = self._w - self.lr * gradient # (d, 1)

            #loss
            loss_value = self.loss_fn(y, y_hat)
            self.losses.append(loss_value)

    def evaluate(self, y, y_hat) -> dict:
        precision = precision_score(y, y_hat, average='binary')
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def sigmoid (self, z:np.ndarray):
        return 1 / (1 + np.exp(-z))
    
    def predict_prob(self, X: np.ndarray):
        z = X @ self._w # (N, 1)
        y_hat = self.sigmoid(z)
        return y_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Trả về nhãn 0 hoặc 1"""
        proba = self.predict_prob(X)
        return (proba >= 0.5).astype(np.int32).ravel()  