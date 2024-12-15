import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#import treefarms
from treefarms import TREEFARMS
from sklearn.datasets import load_wine

print("hello\n")

# Load the Iris dataset
iris = load_wine()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="class")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the TREEFARMS model
config = {
    "regularization": 0.2,  # Penalize trees with more leaves for sparsity
    "rashomon_bound_multiplier": 0.005,  # Controls the size of the Rashomon set
}

# Initialize the TREEFARMS model
#model = treefarms.TREEFARMS(config)
model = TREEFARMS(config)

model.fit(X_train, y_train)

n_trees = model.get_tree_count()
print(f"Number of trees in the Rashomon set: {n_trees}")

# Access the first tree in the Rashomon set
first_tree = model[0]

# Evaluate the first tree on the training data
train_acc = first_tree.score(X_train, y_train)
print(f"Training accuracy of the first tree: {train_acc}")

# Evaluate the first tree on the test data
test_acc = first_tree.score(X_test, y_test)
print(f"Test accuracy of the first tree: {test_acc}")

print("Structure of the first tree:")
print(first_tree)

print("Visualizing the Rashomon set...")
#model.visualize(
#    feature_names=iris.feature_names,
#    feature_description={name: {"info": name, "type": "count", "short": name[:10]} for name in iris.feature_names},
#    width=700,
#    height=500,
#)

# Get predictions on the training data
train_predictions = first_tree.predict(X_train)
print("Predictions on the training data:")
print(train_predictions)

# Get predictions on the test data
test_predictions = first_tree.predict(X_test)
print("Predictions on the test data:")
print(test_predictions)

# Check available methods of the TREEFARMS model
print("Methods and attributes of the TREEFARMS model:")
print(dir(model))

# Check available methods of the first tree
print("Methods and attributes of the first tree:")
print(dir(first_tree))
