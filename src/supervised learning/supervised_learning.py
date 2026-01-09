import math
from collections import Counter

import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Load the extracted features dataset
data = pd.read_csv('data/extracted_features/hand_landmarks.csv') # Replace with actual path to your CSV file

# Separate features and labels
X = data.drop(columns=['instance_id', 'label']).values
y = data['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# First split off test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)

# Then split the remaining data into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=2
)

print(f"Training size: {len(X_train)}") # ~60%
print(f"Validation size: {len(X_val)}") # ~20%
print(f"Testing size: {len(X_test)}") # ~20%
print(f"Total Dataset size: {len(X)}") # 100%

#--------------------------------------------------------------------------------------------
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

print("Model: Decision Tree Classifier")

# Min sample split and max depth act as stopping conditions for the tree growth
param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Load the extracted features dataset 
dt_clf = DecisionTreeClassifier(random_state=2)

# Splits training data into 5 pieces
# Trains model on 4 pieces and tests on the 5th piece
# Repeats this for every combination for hyper parameters in the grid
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Identify best parameters from the grid search
print(f"Best Parameters: {grid_search.best_params_}")
best_tree = grid_search.best_estimator_
# Explain why these parameters were chosen (Highest Mean cross validation accuracy)

# Evaluate the best model on the test set
y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred)) # This gives you Accuracy and Sensitivity
#--------------------------------------------------------------------------------------------
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Model: Random Forrest Classifier")

# Parameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize Random Forest Classifier (random state = 2 for reproducibility)
rf_clf = RandomForestClassifier(random_state=2)

# Grid Search with Cross-Validation
# Trains model on 4 pieces and tests on the 5th piece
# Repeats this for every combination for hyper parameters in the grid
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print (f"Best Parameters: {grid_search.best_params_}")
best_forest = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_forest.predict(X_test)
print(classification_report(y_test, y_pred)) # This gives you Accuracy and Sensitivity
#--------------------------------------------------------------------------------------------