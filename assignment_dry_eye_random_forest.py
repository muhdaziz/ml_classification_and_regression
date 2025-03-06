import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Dry_Eye_Dataset.csv")

X = data[["Age", "Sleep duration", "Sleep quality", "Stress level", "Heart rate", "Daily steps", "Physical activity", "Height", "Weight", "Average screen time"]].values
y = data["Gender"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train.shape, y_test.shape
print(data.shape)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred.shape

class_names = ["M", "F"]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, target_names=class_names)
print("Classification Report:\n", classification_rep)