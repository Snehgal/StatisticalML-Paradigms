# SML - Simple Machine Learning Library

A lightweight Python library for machine learning algorithms and utilities.

## Installation

```bash
pip install sml
```

## Modules

- `Trees` - Decision tree-based models and ensembles
- Additional modules coming soon...

## Trees Module

The `Trees` module provides implementations of decision tree algorithms and ensemble methods.

### Usage

```python
from SML import Trees

# Example: Create and train a decision tree
X_train = [[2.5, 1.0], [3.0, 2.1], [1.5, 3.4], [4.2, 2.5]]  # Feature data
y_train = ["Yes", "No", "Yes", "No"]  # Target labels

# Create a decision tree with max depth 3 and min 1 sample per leaf
tree = Trees.BinaryDecisionTree(maxDepth=3, minSamples=1)
tree.train(X_train, y_train)

# Make predictions
X_test = [[2.0, 1.5], [3.5, 2.8]]
predictions = tree.predict(X_test)
```

## API Reference

### Trees.gini(y)

Calculates the Gini impurity for a set of labels.

**Parameters:**
- `y` (array-like): Array of class labels.

**Returns:**
- `float`: Gini impurity value. A perfect split will have a value of 0.

**Example:**
```python
from SML import Trees
import numpy as np

labels = np.array(["Yes", "No", "Yes", "Yes"])
impurity = Trees.gini(labels)
print(f"Gini impurity: {impurity}")  # Should return 0.375
```

### Trees.findBestSplit(X, y)

Finds the best feature and threshold to split the data based on Gini impurity.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix of shape (d, n) where d is samples and n is features.
- `y` (array-like): Target labels array.

**Returns:**
- `tuple`: (bestFeature, bestThreshold) indices for the optimal split.

**Example:**
```python
from SML import Trees
import numpy as np

X = np.array([[2.5, 1.0], [3.0, 2.1], [1.5, 3.4], [4.2, 2.5]])
y = np.array(["Yes", "No", "Yes", "No"])
feature, threshold = Trees.findBestSplit(X, y)
print(f"Best split: Feature {feature} with threshold {threshold}")
```

### Trees.findBestSplitRF(X, y, numFeatures=2)

Finds the best split for Random Forest by considering only a random subset of features.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix.
- `y` (array-like): Target labels array.
- `numFeatures` (int, optional): Number of features to consider for the split. Default is 2.

**Returns:**
- `tuple`: (bestFeature, bestThreshold) indices for the optimal split.

### Trees.foldSplit(data, k=5)

Splits data into k folds for cross-validation.

**Parameters:**
- `data` (pandas.DataFrame or numpy.ndarray): The dataset to split.
- `k` (int, optional): Number of folds. Default is 5.

**Returns:**
- `list`: A list of k arrays containing indices for each fold.

### Trees.polyFeatures(x, degree)

Creates polynomial features from a single feature vector.

**Parameters:**
- `x` (numpy.ndarray): Input feature vector.
- `degree` (int): Maximum polynomial degree.

**Returns:**
- `numpy.ndarray`: Matrix with polynomial features [x^0, x^1, ..., x^degree].

### Trees.crossValError(data, degree, k=5)

Performs k-fold cross-validation for polynomial regression and returns average MSE.

**Parameters:**
- `data` (pandas.DataFrame): Dataset containing 'x' and 'y' columns.
- `degree` (int): Polynomial degree to use in the regression.
- `k` (int, optional): Number of folds for cross-validation. Default is 5.

**Returns:**
- `float`: Average Mean Squared Error across all folds.

**Example:**
```python
from SML import Trees
import pandas as pd
import numpy as np

# Generate synthetic data
x = np.linspace(0, 10, 100)
y = 2*x + 3*x**2 + np.random.normal(0, 5, 100)
data = pd.DataFrame({'x': x, 'y': y})

# Find optimal polynomial degree using cross-validation
for degree in range(1, 5):
    mse = Trees.crossValError(data, degree=degree, k=5)
    print(f"Degree {degree} - Cross-validation MSE: {mse:.4f}")
```

### Trees.BinaryDecisionTree

A binary decision tree classifier for binary classification problems.

#### Constructor

```python
Trees.BinaryDecisionTree(maxDepth, minSamples, randomForest=False)
```

**Parameters:**
- `maxDepth` (int): Maximum depth of the tree.
- `minSamples` (int): Minimum number of samples required to split a node.
- `randomForest` (bool, optional): If True, uses random feature selection. Default is False.

#### Methods

- `train(X, y)`: Train the decision tree model.
- `predict(X)`: Predict class labels for multiple instances.
- `predictSingle(x)`: Predict the class label for a single instance.
- `printTree()`: Visualize the tree structure.

**Example:**
```python
from SML import Trees
import numpy as np

# Prepare data
X = np.array([[2.5, 1.0], [3.0, 2.1], [1.5, 3.4], [4.2, 2.5]])
y = np.array(["Yes", "No", "Yes", "No"])

# Create and train a decision tree
tree = Trees.BinaryDecisionTree(maxDepth=3, minSamples=1)
tree.train(X, y)

# Visualize the tree
tree.printTree()

# Make predictions
X_test = np.array([[2.0, 1.5], [3.5, 2.8]])
predictions = tree.predict(X_test)
print(f"Predictions: {predictions}")
```

### Trees.BaggedTrees

Implementation of Bootstrap Aggregating (Bagging) for decision trees. Can be configured as a Random Forest.

#### Constructor

```python
Trees.BaggedTrees(numTrees=10, maxDepth=3, minSamples=2, randomForest=False)
```

**Parameters:**
- `numTrees` (int, optional): Number of trees in the ensemble. Default is 10.
- `maxDepth` (int, optional): Maximum depth for each tree. Default is 3.
- `minSamples` (int, optional): Minimum samples required to split a node. Default is 2.
- `randomForest` (bool, optional): If True, uses random feature selection. Default is False.

#### Methods

- `fit(X, y)`: Train the ensemble model.
- `predict(X)`: Predict class labels using majority voting.
- `oobError(y)`: Compute Out-of-Bag error rate.
- `printTrees()`: Print the structure of all trees in the ensemble.

**Example:**
```python
from SML import Trees
import numpy as np

# Prepare data
X = np.array([[2.5, 1.0], [3.0, 2.1], [1.5, 3.4], [4.2, 2.5], 
               [2.2, 2.8], [3.1, 1.2], [1.8, 3.0], [4.5, 2.2]])
y = np.array(["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"])

# Create and train a Random Forest (Bagged Trees with random feature selection)
forest = Trees.BaggedTrees(numTrees=10, maxDepth=3, minSamples=1, randomForest=True)
forest.fit(X, y)

# Compute OOB error
oob_error = forest.oobError(y)
print(f"Out-of-Bag Error: {oob_error:.4f}")

# Make predictions
X_test = np.array([[2.0, 1.5], [3.5, 2.8]])
predictions = forest.predict(X_test)
print(f"Ensemble predictions: {predictions}")
```

## Example Applications

### Classification with Decision Trees

```python
from SML import Trees
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate a non-linear dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = np.where(y == 1, "Yes", "No")  # Convert to string labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree
tree = Trees.BinaryDecisionTree(maxDepth=5, minSamples=2)
tree.train(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
```

### Random Forest Ensemble

```python
from SML import Trees
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load a dataset
data = load_breast_cancer()
X, y = data.data, np.where(data.target == 1, "Malignant", "Benign")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest
forest = Trees.BaggedTrees(numTrees=100, maxDepth=4, minSamples=5, randomForest=True)
forest.fit(X_train, y_train)

# Make predictions
predictions = forest.predict(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Random Forest Accuracy: {accuracy:.4f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.