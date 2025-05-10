# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data: Import the dataset and inspect column names. 

2.Prepare Data: Separate features (X) and target (y).

3.Split Data: Divide into training (80%) and testing (20%) sets.

4.Scale Features: Standardize the data using StandardScaler.

5.Train Model: Fit a Logistic Regression model on the training data.

6.Make Predictions: Predict on the test set. 

7.Evaluate Model: Calculate accuracy, precision, recall, and classification report. 

8.Confusion Matrix: Compute and visualize confusion matrix.


## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: GANESH PRABHU J
RegisterNumber: 212223220023
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Select CSV file using file dialog
tk.Tk().withdraw()  # Hide the root window
file_path = askopenfilename(title="Select food_items.csv file", filetypes=[("CSV Files", "*.csv")])
data = pd.read_csv(file_path)

# Print column names
print("Column Names in the Dataset:")
print(data.columns)

# Separate features (X) and target (y)
X = data.drop(columns=['class'])  # Nutritional information as features
y = data['class']  # Target: class labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model with increased max_iter
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict the classifications on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model for multiclass classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
evaluation_report = classification_report(y_test, y_pred)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")
print("\nClassification Report:\n", evaluation_report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Output:

![WhatsApp Image 2025-05-10 at 12 42 11_66d4fac1](https://github.com/user-attachments/assets/9f96ebfe-84ec-45cc-a70c-68908a0b916a)


![WhatsApp Image 2025-05-10 at 12 42 20_040d189d](https://github.com/user-attachments/assets/1a7b1830-3a20-4ad9-bdeb-0734b1482f54)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
