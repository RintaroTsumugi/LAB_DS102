import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load data
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

# Filter 0 & 1
train = np.isin(train_labels, [0,1])
test = np.isin(test_labels, [0,1])

#flatten images
N = np.sum(train)
train_images = train_images[train].reshape(N, -1).astype(np.float64) / 255.0
N = np.sum(test)
test_images = test_images[test].reshape(N, -1).astype(np.float64) / 255.0

train_labels = train_labels[train]
test_labels = test_labels[test]

logreg = LogisticRegression(random_state=16)

logreg.fit(train_images, train_labels)

labels_pred = logreg.predict(test_images)

# Classification_report
target_names = ['labels_0', 'labels_1']
print(classification_report(test_labels, labels_pred, target_names=target_names))

# Evaluation Train Test
acc = accuracy_score(test_labels, labels_pred)
prec = precision_score(test_labels, labels_pred, average='macro')
rec = recall_score(test_labels, labels_pred, average='macro')
f1_macro = f1_score(test_labels, labels_pred, average='macro')

train_acc = accuracy_score(train_labels, logreg.predict(train_images))
train_prec = precision_score(train_labels, logreg.predict(train_images), average='macro')
train_rec = recall_score(train_labels, logreg.predict(train_images), average='macro')
train_f1_macro = f1_score(train_labels, logreg.predict(train_images), average='macro')

print(f"Train Evaluation    : Accuracy: {train_acc:.2f} , Precision (macro): {train_prec:.2f} , Recall (macro): {train_rec:.2f} , F1-score (macro): {train_f1_macro:.2f}")
print(f"Test  Evaluation   : Accuracy: {acc:.2f} , Precision (macro): {prec:.2f} , Recall (macro){rec:.2f} , F1-score (macro): {f1_macro:.2f}")

