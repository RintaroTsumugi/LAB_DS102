import cv2 as cv
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

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

X_train, y_train = shuffle(train_data['images'], train_data['labels'], random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data['images'])


# Train model
model = LinearSVC(
    C=0.001, 
    dual=False, 
    max_iter=150, 
    class_weight='balanced', # Giúp cân bằng Recall và Precision
    random_state=42
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(y_pred)

precision = precision_score(test_data['labels'], y_pred)
recall = recall_score(test_data['labels'], y_pred)
f1 = f1_score(test_data['labels'], y_pred)

print(f"Precision_score: {precision}")
print(f"Recall_score: {recall}")
print(f"f1_score: {f1}")

# So sánh vs SVM ở assignment 1
manual_res = [0.823, 0.675, 0.742]
library_res = [precision, recall, f1] 
metrics = ["Precision", "Recall", "F1-score"]

print("\n" + "="*50)
print(f"{'Metric':<15} | {'Implemented SVM':<15} | {'Library SVM':<15}")
print("-" * 50)

for i in range(len(metrics)):
    print(f"{metrics[i]:<15} | {manual_res[i]:<15.4f} | {library_res[i]:<15.4f}")

print("="*50 + "\n")