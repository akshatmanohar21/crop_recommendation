import pandas as pd
import numpy as np
from random import seed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Crop_recommendation.csv")

features = data.drop('label', axis=1)
target = data['label']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=seed)

model = RandomForestClassifier(random_state=seed)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(predictions, y_test)
print(f"Random Forest Model Accuracy: {accuracy:.4f}")

train_predictions = model.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_predictions):.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

confusion_matrix_result = confusion_matrix(y_test, predictions)
plt.figure(figsize=(15, 15))
sns.heatmap(confusion_matrix_result, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
title = f'Confusion Matrix - Accuracy: {accuracy:.4f}'
plt.title(title, size=15)
plt.show()

print(classification_report(y_test, predictions))

new_data = np.array([[90, 42, 43, 20.879744, 75, 5.5, 220]])
predicted_crop = model.predict(new_data)
print(f"Predicted Crop:", predicted_crop[0])

with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nRequired Dependencies:")
for module in set([
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"
]):
    print(f"{module}")
