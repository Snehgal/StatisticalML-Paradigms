# SML - Simple Machine Learning Library

A lightweight Python library for machine learning algorithms and utilities.

## Table of Contents

- [Installation](#installation)
- [Modules Overview](#modules)
- [Trees Module](#trees-module)
  <details>
    <summary>Classes and Functions</summary>

    - [BinaryDecisionTree](#treesbinarydecisiontree)
    - [BaggedTrees](#treesbagged-trees)
    - [gini](#treesgini)
    - [findBestSplit](#treesfindbestsplit)
    - [findBestSplitRF](#treesfindbestsplitrf)
    - [foldSplit](#treesfoldsplit)
    - [polyFeatures](#treespolyfeatures)
    - [crossValError](#treescrossvalerror)
  </details>
- [DataPreprocessing Module](#datapreprocessing-module)
  <details>
    <summary>Functions</summary>
    
    - [numpy_split](#datapreprocessingnumpy_split)
    - [accuracy](#datapreprocessingaccuracy)
    - [pca](#datapreprocessingpca)
    - [FDA](#datapreprocessingfda)
  </details>
- [Regression Module](#regression-module)
  <details>
    <summary>Classes and Functions</summary>
    
    - [DecisionStump](#regressiondecisionstump)
    - [GradientBoosting](#regressiongradientboosting)
    - [SimpleNeuralNetwork](#regressionsimpleneuralnetwork)
    - [gaussian](#regressiongaussian)
    - [predictMLE](#regressionpredictmle)
    - [discriminantLDA](#regressiondiscriminantlda)
    - [predictLDA](#regressionpredictlda)
    - [discriminantQDA](#regressiondiscriminantqda)
    - [predictQDA](#regressionpredictqda)
    - [PCAthenLDA](#regressionpcathenlda)
  </details>
- [Example Applications](#example-applications)
  <details>
    <summary>Examples</summary>
    
    - [Classification with Decision Trees](#classification-with-decision-trees)
    - [Random Forest Ensemble](#random-forest-ensemble)
    - [Dimensionality Reduction with PCA](#dimensionality-reduction-with-pca)
    - [Regression with Gradient Boosting](#regression-with-gradient-boosting)
    - [Neural Network for XOR Classification](#neural-network-for-xor-classification)
  </details>
- [License](#license)

## Installation

```bash
1. Download and extract file or clone repo
2. Rename the fodler to SML
3. Place it in working directory 
4. Follow documentation
```

## Modules

- `Trees` - Decision tree-based models and ensembles
- `DataPreprocessing` - Data splitting, processing, and dimensionality reduction utilities
- `Regression` - Gradient boosting and neural network implementations
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

### `Trees.BinaryDecisionTree`

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

### `Trees.Bagged Trees`

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

### `Trees.gini`

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

### `Trees.findBestSplit`
`
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

### `Trees.findBestSplitRF`

Finds the best split for Random Forest by considering only a random subset of features.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix.
- `y` (array-like): Target labels array.
- `numFeatures` (int, optional): Number of features to consider for the split. Default is 2.

**Returns:**
- `tuple`: (bestFeature, bestThreshold) indices for the optimal split.

### `Trees.foldSplit`

Splits data into k folds for cross-validation.

**Parameters:**
- `data` (pandas.DataFrame or numpy.ndarray): The dataset to split.
- `k` (int, optional): Number of folds. Default is 5.

**Returns:**
- `list`: A list of k arrays containing indices for each fold.

### `Trees.polyFeatures`

Creates polynomial features from a single feature vector.

**Parameters:**
- `x` (numpy.ndarray): Input feature vector.
- `degree` (int): Maximum polynomial degree.

**Returns:**
- `numpy.ndarray`: Matrix with polynomial features [x^0, x^1, ..., x^degree].

### `Trees.crossValError`

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

## DataPreprocessing Module

The `DataPreprocessing` module provides utilities for data preprocessing, including data splitting, accuracy calculation, and dimensionality reduction techniques like PCA and FDA.

### Usage

```python
from SML import DataPreprocessing
import numpy as np

# Example: Split data into training and test sets
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])
X_train, X_test, y_train, y_test = DataPreprocessing.numpy_split(X, y, test_size=0.2, random_state=42)

# Example: Perform PCA on a dataset
X = np.random.randn(10, 100)  # 10 features, 100 samples
X_reduced, U_p, p = DataPreprocessing.pca(X, preservedVariance=0.95)
```

### `DataPreprocessing.numpy_split`

Split dataset into train and test sets.

**Parameters:**
- `x` (np.ndarray): Features (n_samples, n_features).
- `y` (np.ndarray): Labels (n_samples,).
- `test_size` (float): Fraction of data to reserve for testing.
- `random_state` (int, optional): Random seed for reproducibility.

**Returns:**
- `tuple`: (x_train, x_test, y_train, y_test) splits of input data.

**Example:**
```python
from SML import DataPreprocessing
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = DataPreprocessing.numpy_split(X, y, test_size=0.2, random_state=42)
```

### `DataPreprocessing.accuracy`

Compute classification accuracy.

**Parameters:**
- `xTest` (np.ndarray): Test features of shape (n_features, n_samples).
- `labels` (np.ndarray): True labels.
- `predict` (function): Function that predicts label given a sample.

**Returns:**
- `float`: Accuracy (correct predictions / total predictions).

**Example:**
```python
from SML import DataPreprocessing, Trees
import numpy as np

# Prepare data
X = np.array([[2.5, 1.0], [3.0, 2.1], [1.5, 3.4], [4.2, 2.5]]).T  # Transpose to match expected format
y = np.array(["Yes", "No", "Yes", "No"])

# Train a model
tree = Trees.BinaryDecisionTree(maxDepth=3, minSamples=1)
tree.train(X.T, y)  # Note: tree.train expects (n_samples, n_features)

# Calculate accuracy
acc = DataPreprocessing.accuracy(X, y, tree.predictSingle)
print(f"Accuracy: {acc:.4f}")
```

### `DataPreprocessing.pca`

Perform Principal Component Analysis (PCA) on the data.

**Parameters:**
- `X` (np.ndarray): Data matrix of shape (features, samples).
- `preservedVariance` (float, optional): Desired variance retention (e.g., 0.95 or 95 for 95%).
- `pcaComponents` (int, optional): Number of PCA components to reduce to.

**Returns:**
- `tuple`: (X_reduced, U_p, p)
  - `X_reduced` (np.ndarray): Reduced data matrix (p components x samples).
  - `U_p` (np.ndarray): Top eigenvectors (features x p components).
  - `p` (int): Number of components selected.

**Example:**
```python
from SML import DataPreprocessing
import numpy as np

# Create a dataset with 10 features and 100 samples
X = np.random.randn(10, 100)

# Reduce dimensionality while preserving 95% of variance
X_reduced, U_p, p = DataPreprocessing.pca(X, preservedVariance=0.95)
print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")

# Or specify exact number of components
X_reduced_2, U_p_2, _ = DataPreprocessing.pca(X, pcaComponents=3)
print(f"Reduced to exactly 3 components: {X_reduced_2.shape}")
```

### `DataPreprocessing.FDA`

Perform Fisher Discriminant Analysis (FDA) on training sets.

**Parameters:**
- `train_sets` (list of np.ndarray): List of datasets, each (features, samples) for a class.
- `delta` (float): Regularization parameter for stability.
- `num_components` (int): Number of FDA components to retain.
- `graph` (bool): Whether to plot the projected data.

**Returns:**
- `tuple`: (W, X_projected)
  - `W` (np.ndarray): Projection matrix (features x num_components).
  - `X_projected` (np.ndarray): Projected dataset (num_components x total_samples).

**Example:**
```python
from SML import DataPreprocessing
import numpy as np

# Create two classes with 5 features and 20 samples each
class1 = np.random.randn(5, 20) + np.array([2, 0, 0, 0, 0]).reshape(-1, 1)
class2 = np.random.randn(5, 20) + np.array([-2, 0, 0, 0, 0]).reshape(-1, 1)

# Perform FDA
W, X_projected = DataPreprocessing.FDA([class1, class2], num_components=2, graph=True)

print(f"Projection matrix shape: {W.shape}")
print(f"Projected data shape: {X_projected.shape}")
```

## Regression Module

The `Regression` module provides implementations of regression algorithms including gradient boosting with decision stumps and a simple neural network.

### Usage

```python
from SML import Regression
import numpy as np

# Example: Train a gradient boosting model for regression
X_train = np.linspace(0, 1, 100).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train.flatten()) + 0.1 * np.random.randn(100)

# Create and train a gradient boosting model
gbm = Regression.GradientBoosting(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)

# Make predictions
X_test = np.linspace(0, 1, 20).reshape(-1, 1)
predictions = gbm.predict(X_test)

# Example: Train a simple neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR function

nn = Regression.SimpleNeuralNetwork()
nn.train(X, y, learning_rate=0.1, epochs=1000, method="gradientDescent")
```

### `Regression.DecisionStump`

A one-level decision tree (decision stump) for regression tasks.

#### Constructor

```python
Regression.DecisionStump()
```

#### Methods

- `fit(X, residuals, n_cuts=20)`: Train the stump to fit residuals by finding the best threshold.
- `predict(X)`: Predict output values for given input features.

**Example:**
```python
from SML import Regression
import numpy as np

# Create simple regression data
X = np.linspace(0, 1, 50).reshape(-1, 1)
y = np.sin(2 * np.pi * X.flatten()) + 0.1 * np.random.randn(50)

# Fit a decision stump to the data
stump = Regression.DecisionStump()
stump.fit(X, y, n_cuts=20)

# Make predictions
predictions = stump.predict(X)
print(f"Decision Stump: {stump}")
```

### `Regression.GradientBoosting`

Implements a Gradient Boosting Machine (GBM) using Decision Stumps as weak learners.

#### Constructor

```python
Regression.GradientBoosting(n_estimators=100, learning_rate=0.01, loss='squared', uniform_cuts=20)
```

**Parameters:**
- `n_estimators` (int, optional): Number of boosting rounds. Default is 100.
- `learning_rate` (float, optional): Learning rate (shrinkage factor). Default is 0.01.
- `loss` (str, optional): Loss function ('squared' or 'absolute'). Default is 'squared'.
- `uniform_cuts` (int, optional): Number of threshold cuts considered when fitting stumps. Default is 20.

#### Methods

- `fit(X, y)`: Train the Gradient Boosting model on the input data.
- `predict(X)`: Predict output values for input X using the trained model.

**Example:**
```python
from SML import Regression
import numpy as np
import matplotlib.pyplot as plt

# Create regression data
X_train = np.linspace(0, 1, 100).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train.flatten()) + 0.1 * np.random.randn(100)

# Create and train a gradient boosting model
gbm = Regression.GradientBoosting(n_estimators=100, learning_rate=0.1, loss='squared')
gbm.fit(X_train, y_train)

# Make predictions
X_test = np.linspace(0, 1, 200).reshape(-1, 1)
predictions = gbm.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label='Training data', alpha=0.5)
plt.plot(X_test, predictions, 'r-', label='GBM predictions', linewidth=2)
plt.title('Gradient Boosting Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### `Regression.SimpleNeuralNetwork`

Implements a simple fully connected neural network with one hidden layer for regression or classification tasks.

#### Constructor

```python
Regression.SimpleNeuralNetwork()
```

#### Methods

- `forward(X)`: Performs forward propagation through the network.
- `compute_loss(y, y_hat)`: Computes mean squared error (MSE) loss.
- `train(X, y, learning_rate=0.1, epochs=1000, method="gradientDescent")`: Trains the neural network.
  - `method`: "gradientDescent" for batch updates or "stochastic" for sample-wise updates.

**Example:**
```python
from SML import Regression
import numpy as np

# Create XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train a neural network
nn = Regression.SimpleNeuralNetwork()
nn.train(X, y, learning_rate=0.1, epochs=1000, method="gradientDescent")

# Make predictions
predictions = nn.forward(X)
print("Neural Network Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, True: {y[i][0]}, Predicted: {predictions[i][0]:.3f}")
```

### `Regression.gaussian`

Computes log probability density of multivariate Gaussian distribution.

**Parameters:**
- `x` (np.ndarray): Input sample.
- `u` (np.ndarray): Mean vector.
- `cov` (np.ndarray): Covariance matrix.

**Returns:**
- `float`: Log probability of sample under Gaussian.

### `Regression.predictMLE`

Predict class label for a sample using Maximum Likelihood Estimation (MLE).

**Parameters:**
- `x` (np.ndarray): Input sample.
- `u0, u1, u2` (np.ndarray): Class means.
- `cov0, cov1, cov2` (np.ndarray): Class covariance matrices.

**Returns:**
- `int`: Predicted class label (0, 1, or 2).

### `Regression.discriminantLDA`

Compute LDA discriminant scores for each class.

**Parameters:**
- `X` (np.ndarray): Input features (shape [features, samples]).
- `us` (list of np.ndarray): List of class means.
- `cov` (np.ndarray): Shared covariance matrix.

**Returns:**
- `np.ndarray`: Discriminant scores for each class.

### `Regression.predictLDA`

Predict class labels using Linear Discriminant Analysis (LDA).

**Parameters:**
- `X` (np.ndarray): Input features (shape [features, samples]).
- `us` (list of np.ndarray): List of class means.
- `cov` (np.ndarray): Shared covariance matrix.

**Returns:**
- `np.ndarray`: Predicted class labels.

### `Regression.discriminantQDA`

Compute QDA discriminant scores for each class.

**Parameters:**
- `X` (np.ndarray): Input features (shape [features, samples]).
- `us` (list of np.ndarray): List of class means.
- `covs` (list of np.ndarray): List of class covariance matrices.
- `priors` (list of float, optional): List of class prior probabilities.

**Returns:**
- `np.ndarray`: Discriminant scores for each class.

### `Regression.predictQDA`

Predict class labels using Quadratic Discriminant Analysis (QDA).

**Parameters:**
- `X` (np.ndarray): Input features (shape [features, samples]).
- `us` (list of np.ndarray): List of class means.
- `covs` (list of np.ndarray): List of class covariance matrices.
- `priors` (list of float, optional): List of class prior probabilities.

**Returns:**
- `np.ndarray`: Predicted class labels.

### `Regression.PCAthenLDA`

Perform Principal Component Analysis (PCA) followed by Linear Discriminant Analysis (LDA) on the dataset.

**Parameters:**
- `presVar` (float): The preserved variance for PCA.
- `numPCA` (int, optional): The number of principal components to retain. Default is 0 (auto-determined by presVar).

**Returns:**
- `tuple`: (predictionsTestPCA_LDA, predictionsTrainPCA_LDA)
  - `predictionsTestPCA_LDA` (np.ndarray): Predicted test set labels.
  - `predictionsTrainPCA_LDA` (np.ndarray): Predicted training set labels.

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

### Dimensionality Reduction with PCA

```python
from SML import DataPreprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data.T  # Transpose to match expected format (features, samples)
y = digits.target

# Apply PCA to reduce to 2 dimensions for visualization
X_reduced, _, _ = DataPreprocessing.pca(X, pcaComponents=2)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[0], X_reduced[1], c=y, cmap='tab10')
plt.colorbar(scatter, label='Digit')
plt.title('PCA projection of digits dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

### Regression with Gradient Boosting

```python
from SML import Regression
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * np.sin(X.flatten()) + 0.1 * X.flatten()**2 + np.random.normal(0, 0.5, 100)

# Split data into train and test sets
indices = np.random.permutation(len(X))
train_size = 70
X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

# Train a gradient boosting model
gbm = Regression.GradientBoosting(n_estimators=200, learning_rate=0.05, loss='squared')
gbm.fit(X_train, y_train)

# Make predictions
predictions = gbm.predict(X_test)

# Calculate R-squared
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, c='r', alpha=0.5, label='Test data')
plt.plot(X_test, predictions, 'g-', linewidth=2, label='GBM predictions')
plt.title(f'Gradient Boosting Regression (R² = {r_squared:.4f})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Plot training loss history
plt.figure(figsize=(10, 6))
plt.plot(gbm.train_loss_history)
plt.xlabel('Iterations')
plt.ylabel('Training Loss')
plt.title('GBM Training Loss over Iterations')
plt.grid(True)
plt.show()
```

### Neural Network for AND/OR/NOR/NAND/XOR/XNOR Classification

```python
from SML import Regression
import numpy as np
import matplotlib.pyplot as plt

func = "NOR"
# Create AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

if func == "AND":
    y = np.array([[0], [0], [0], [1]])
if func == "OR":
    y = np.array([[0], [1], [1], [1]])
if func == "NAND":
    y = np.array([[1], [1], [1], [0]])
if func == "NOR":
    y = np.array([[1], [0], [0], [0]])
if func == "XNOR":
    y = np.array([[1], [0], [0], [1]])
if func == "XOR":
    y = np.array([[0], [1], [1], [0]])


# Train neural network
nn = Regression.SimpleNeuralNetwork()
nn.train(X, y, learning_rate=0.1, epochs=2000, method="gradientDescent")

# Create a meshgrid for visualization
h = 0.01
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Make predictions on the grid
Z = np.zeros(grid_points.shape[0])
for i, point in enumerate(grid_points):
    Z[i] = nn.forward(point.reshape(1, -1))[0, 0]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

# Use a simpler colormap
# plt.contourf(xx, yy, Z, alpha=0.6, cmap='coolwarm')
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

# Plot the points clearly
colors = ['blue' if label == 0 else 'orange' for label in y.flatten()]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=100, label='Data Points')

# Manually add markers for class labels
for i, txt in enumerate(y.flatten()):
    plt.annotate(f'Class {txt}', (X[i, 0] + 0.02, X[i, 1] + 0.02), fontsize=12)

# Add titles and labels
plt.title(f'Simple Neural Network Learning {func}', fontsize=16)
plt.xlabel('Input Feature 1', fontsize=14)
plt.ylabel('Input Feature 2', fontsize=14)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', edgecolor='k', label='Class 0'),
                   Patch(facecolor='orange', edgecolor='k', label='Class 1')]
plt.legend(handles=legend_elements, fontsize=12)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

## License

This project is not under licensing yet.