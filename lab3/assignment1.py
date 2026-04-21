import cv2 as cv
import os
from svm import SVM
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
BASE_DIR = 'chest_xray'

def load_data(split):
    normal = "NORMAL"
    pneumonia  = 'PNEUMONIA'
    
    images = []
    labels = []

    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):   
        image = cv.imread(os.path.join(BASE_DIR, split, normal, img_file))
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, (128,128), interpolation=cv.INTER_NEAREST)
        image = image.flatten()
        images.append(image)
        labels.append(1)

    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneumonia)):
        image = cv.imread(os.path.join(BASE_DIR, split, pneumonia, img_file))
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, (128,128), interpolation=cv.INTER_NEAREST)
        image = image.flatten()
        images.append(image)
        labels.append(-1)

    images = np.stack(images, axis = 0)
    return {
        "images": images, 
        "labels": np.array(labels)
    }

# Load dữ liệu
train_data = load_data(split='train')
test_data = load_data(split='test')

# Chuẩn hóa Z_score
X_train = train_data['images']
X_test = test_data['images']

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)


train_data['images'] = (X_train - X_mean) / X_std
test_data['images'] = (X_test - X_mean) / X_std

# Train model
model = SVM(0.01)
model.fit(train_data['images'], train_data['labels'])
y_pred = model.predict(test_data['images'])
print(y_pred)
y_pred = np.where(y_pred >= -0, 1, -1)
print(y_pred)

precision = precision_score(test_data['labels'], y_pred)
recall = recall_score(test_data['labels'], y_pred)
f1 = f1_score(test_data['labels'], y_pred)

print(f"Precision_score: {precision}")
print(f"Recall_score: {recall}")
print(f"f1_score: {f1}")

plt.figure(figsize=(10, 6))
plt.plot(model.losses, label='Hinge Loss', color='#2c3e50', linewidth=2)
plt.title('SVM Training Convergence', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()