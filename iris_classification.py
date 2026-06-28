# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load dataset
iris = load_iris()

X = iris.data   # input (length, width)
y = iris.target # output (flower type)

# Step 3: Split data (train & test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LogisticRegression(max_iter=200)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Accuracy check
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 9: Try your own input
sample = [[5.1, 3.5, 1.4, 0.2]]  # example
prediction = model.predict(sample)

print("\nPredicted Flower:", iris.target_names[prediction][0])