import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

#load dataset
iris = load_iris()
x = iris.data
y = iris.target

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#train models
lr = LogisticRegression()
lr.fit(x_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

#predictions
lr_pred = lr.predict(x_test)
knn_pred = knn.predict(x_test)
dt_pred = dt.predict(x_test)

#accuracy
lr_acc = accuracy_score(y_test, lr_pred)
knn_acc = accuracy_score(y_test, knn_pred)
dt_acc = accuracy_score(y_test, dt_pred)

#results
results = pd.DataFrame({"Model": ["Logistic Regression", "KNN", "Decision Tree"], "Accuracy": [lr_acc, knn_acc, dt_acc]})
print(results)

#confusion matrix
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
