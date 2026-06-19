from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

iris = load_iris()

X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/iris_model.joblib")

print("Model saved successfully!")
