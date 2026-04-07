import numpy as np
from tqdm import tqdm
import idx2numpy
from sklearn.metrics import precision_score, recall_score, f1_score
from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt

# Load data
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
test_data = (test_images, test_labels)

def filter_data(data, condition):
    images, labels = data

    new_images = images[labels == condition]
    new_labels = labels[labels == condition]

    return new_images, new_labels

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1)

train_images = np.concatenate(
    [train_images_0, train_images_1],
    axis = 0
)

train_labels = np.concatenate(
    [train_labels_0, train_labels_1],
    axis = 0
)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)

test_images = np.concatenate(
    [test_images_0, test_images_1],
    axis = 0
)

test_labels = np.concatenate(
    [test_labels_0, test_labels_1],
    axis = 0
)

# Flatten images
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1).astype(np.float64) # N (28, 28)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1).astype(np.float64) # N (28, 28)

# Normalize pixel values to [0,1]
train_images /= 255.0
test_images /= 255.0

# Train model
model = LogisticRegression(epoch = 100, lr = 0.1)

model.fit(train_images, train_labels)

# Evaluate on Train set
predict_train_labels = model.predict(train_images)
train_eval = model.evaluate(train_labels,predict_train_labels)
print(f"Train Evaluation: {train_eval}")

# Evaluate on Test set
predict_labels = model.predict(test_images)
test_eval = model.evaluate(test_labels,predict_labels)
print(f"Test Evaluation: {test_eval}")

train_losses = model.losses

# Visualize training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='b', label='Training Loss')
plt.title('Quá trình thay đổi của Hàm Mất Mát (Loss Function)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()