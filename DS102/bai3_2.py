import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# Flatten and normalize
N, _, _ = train_images.shape
train_images = train_images.reshape(N, -1).astype(np.float64) / 255.0 # N (28, 28)
N, _, _ = test_images.shape
test_images = test_images.reshape(N, -1).astype(np.float64) / 255.0 # N (28, 28)

softmax_reg = LogisticRegression(
    solver='lbfgs',
    max_iter=500,
    random_state=16
)

softmax_reg.fit(train_images, train_labels)

labels_pred = softmax_reg.predict(test_images)

target_names = ['labels_0', 'labels_1', 'labels_2','labels_3','labels_4','labels_5','labels_6','labels_7','labels_8','labels_9']
print(classification_report(test_labels, labels_pred, target_names=target_names))

# Evaluation Train Test
acc = accuracy_score(test_labels, labels_pred)
prec = precision_score(test_labels, labels_pred, average='macro')
rec = recall_score(test_labels, labels_pred, average='macro')
f1_macro = f1_score(test_labels, labels_pred, average='macro')

train_acc = accuracy_score(train_labels, softmax_reg.predict(train_images))
train_prec = precision_score(train_labels, softmax_reg.predict(train_images), average='macro')
train_rec = recall_score(train_labels, softmax_reg.predict(train_images), average='macro')
train_f1_macro = f1_score(train_labels, softmax_reg.predict(train_images),average='macro')

print(f"Train Evaluation    : Accuracy: {train_acc:.2f} , Precision (macro): {train_prec:.2f} , Recall (macro): {train_rec:.2f} , F1-score (macro): {train_f1_macro:.2f}")
print(f"Test  Evaluation   : Accuracy: {acc:.2f} , Precision (macro): {prec:.2f} , Recall (macro){rec:.2f} , F1-score (macro): {f1_macro:.2f}")
