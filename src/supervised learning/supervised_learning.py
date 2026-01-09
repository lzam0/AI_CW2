import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Models
from knn_from_scratch import KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# SkLearn Analyse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load the extracted features dataset
cleaned_csv = "data/extracted_features/hand_landmarks_sanitised.csv"
df = pd.read_csv(cleaned_csv)

# Separate features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Create training testing set - random state utilised so that the split is reproducible (seed)
# 60% Train (to learn), 20% Val (to tune), 20% Test (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2
)

print(f"Dataset Split: {len(X_train)} training samples, {len(X_test)} testing samples.")

#--------------------------------------------------------------------------------------------
# Train and evaluate the KNN model

print("Model: kNN Classifier")

# Use the best model selected (k=3 and distance = euclidean)
knn_model = KNN(k=3, distance_metric='euclidean')
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Calculate Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Test Accuracy (k=3, Euclidean): {test_accuracy:.4%}")

# Use sklearn classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#--------------------------------------------------------------------------------------------
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

print("Model: Random Forest Classifier")

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

# Plot Confusion Matrix
def plot_cm(y_true, y_pred, model_name):

    # Calculate the accuracy
    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f'Confusion Matrix: {model_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'src/supervised learning/{model_name}.png')
    plt.show()

# kNN
y_pred_knn = knn_model.predict(X_test) 
plot_cm(y_test, y_pred_knn, "kNN")

# Decision Tree
y_pred_tree = best_tree.predict(X_test)
plot_cm(y_test, y_pred_tree, "Decision Tree")

# Random Forest
y_pred_forest = best_forest.predict(X_test)
plot_cm(y_test, y_pred_forest, "Random Forest")

#--------------------------------------------------------------------------------------------
# Identify the BEST classification model

# Store accuracies for comparison
results = {
    "kNN": accuracy_score(y_test, y_pred_knn),
    "Decision Tree": accuracy_score(y_test, y_pred_tree),
    "Random Forest": accuracy_score(y_test, y_pred_forest)
}

# Present the best model
best_model_name = max(results, key=results.get)
print(f"\n--- Final Analysis ---")
for model, acc in results.items():
    print(f"{model} Accuracy: {acc:.4%}")

print(f"\nThe best classifier is: {best_model_name}")