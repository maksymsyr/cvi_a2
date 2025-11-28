import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------
# LOAD DATA
# -------------------------
train_data = pd.read_csv("Q2/mnist_train.csv")
test_data = pd.read_csv("Q2/mnist_test.csv")

X_train = train_data.iloc[:, 1:].values / 255.0
y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values / 255.0
y_test = test_data.iloc[:, 0].values

# -------------------------
# SCALE DATA
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# KNN CLASSIFIER
# -------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, pred_knn))
joblib.dump(knn, "mnist_knn_model.z")
print("[INFO] KNN model saved as 'mnist_knn_model.z'")

# -------------------------
# LOGISTIC REGRESSION MODEL
# -------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

predictions_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions_logreg))

joblib.dump(logreg, "mnist_logreg_model.z")
print("[INFO] Logistic Regression model saved as 'mnist_logreg_model.z'")
