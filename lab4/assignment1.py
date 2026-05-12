import pandas as pd
from decision_tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

red_wine = pd.read_csv("winequality-red.csv", sep = ";")
white_wine = pd.read_csv("winequality-white.csv", sep = ";")

red_wine["type"] = 0 # rượu đỏ là loại 0
white_wine["type"] = 1 # rượu trắng là loại 1

wine = pd.concat([red_wine,white_wine]).reset_index(drop=True)

X = wine.drop("quality", axis=1).values
y = wine["quality"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Decision Tree - f1 score: {f1}")