import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error
from DataProcessing import pca,numpy_split

class BinaryDecisionTree:
    """
    A binary decision tree classifier for binary classification problems.
    
    This class implements a decision tree with recursive binary splits based on
    feature thresholds that minimize Gini impurity.
    
    Attributes:
        maxDepth (int): Maximum depth of the tree.
        minSamples (int): Minimum number of samples required to split a node.
        randomForest (bool): If True, uses random feature selection for splitting (for Random Forest).
        tree (dict): The trained decision tree structure.
    """
    
    def __init__(self, maxDepth, minSamples, randomForest=False):
        """
        Initialize a new BinaryDecisionTree.
        
        Parameters:
            maxDepth (int): Maximum depth of the tree.
            minSamples (int): Minimum number of samples required to split a node.
            randomForest (bool, optional): If True, uses random feature selection. Default is False.
        """
        self.maxDepth = maxDepth
        self.minSamples = minSamples
        self.randomForest = randomForest
        self.tree = None

    def fit(self, X, y, depth=0, parent_leaf_prediction=None):
        """
        Recursively builds the decision tree.
        
        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (array-like): Target labels.
            depth (int, optional): Current depth in the tree. Default is 0.
            parent_leaf_prediction (str, optional): Prediction from the parent node.
                                                   Used for handling ties. Default is None.
        
        Returns:
            dict: A node in the decision tree, either a leaf with a prediction or
                 an internal node with split information and child nodes.
        """
        # Stopping conditions: max depth, all samples in the same class, or too few samples
        if depth == self.maxDepth or len(set(y)) == 1 or len(y) <= self.minSamples:
            prediction = max(set(y), key=list(y).count)
            
            # Handle the case where there's a tie in votes at the leaf node
            if len(set(y)) == 2 and prediction == parent_leaf_prediction:
                # If there's a tie, flip the prediction (binary tree)
                if prediction == "Yes":
                    prediction = "No"
                elif prediction == "No":
                    prediction = "Yes"
            
            return {"leaf": True, "prediction": prediction}
    
        if self.randomForest:
            bestFeature, bestThreshold = findBestSplitRF(X, y)
        else:
            bestFeature, bestThreshold = findBestSplit(X, y)
    
        if bestFeature is None:
            # When no split is found (could happen when all samples are homogeneous)
            prediction = max(set(y), key=list(y).count)
            if len(set(y)) == 2 and prediction == parent_leaf_prediction:
                # Flip prediction in case of a tie
                if prediction == "Yes":
                    prediction = "No"
                elif prediction == "No":
                    prediction = "Yes"
            return {"leaf": True, "prediction": prediction}
    
        leftSplit = X[:, bestFeature] <= bestThreshold            
        rightSplit = X[:, bestFeature] > bestThreshold
    
        # Recursively create left and right subtrees
        leftNode = self.fit(X[leftSplit], y[leftSplit], depth + 1, parent_leaf_prediction)
        rightNode = self.fit(X[rightSplit], y[rightSplit], depth + 1, parent_leaf_prediction)
    
        # Calculate the majority vote for this node
        majorityVote = max(set(y), key=list(y).count)
    
        # Check if both left and right subtrees are leaf nodes with the same prediction
        if leftNode["leaf"] and rightNode["leaf"]:
            if leftNode["prediction"] == rightNode["prediction"]:
                # If both predictions are the same, flip the one that's not in the majority
                if leftNode["prediction"] != majorityVote:
                    leftNode["prediction"] = "Yes" if majorityVote == "No" else "No"
                else:
                    rightNode["prediction"] = "Yes" if majorityVote == "No" else "No"
    
        return {
            "leaf": False,
            "feature": bestFeature,
            "threshold": bestThreshold,
            "left": leftNode,
            "right": rightNode
        }

    def train(self, X, y):
        """
        Train the decision tree model.
        
        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (array-like): Target labels.
        """
        self.tree = self.fit(X, y, 0)

    def predictSingle(self, x):
        """
        Predict the class label for a single instance.
        
        Parameters:
            x (numpy.ndarray): A single feature vector.
            
        Returns:
            str: The predicted class label.
        """
        current = self.tree
        
        while not current["leaf"]:
            if x[current["feature"]] <= current["threshold"]:
                current = current["left"]
            else:
                current = current["right"]

        return current["prediction"]

    def predict(self, X):
        """
        Predict class labels for multiple instances.
        
        Parameters:
            X (numpy.ndarray): Feature matrix where each row is an instance.
            
        Returns:
            numpy.ndarray: Array of predicted class labels.
        """
        return np.array([self.predictSingle(x) for x in X])

    def printTree(self, node=None, depth=0):
        """
        Recursively prints the tree structure for visualization.
        
        Parameters:
            node (dict, optional): Current node to print. Default is None (starts from the root).
            depth (int, optional): Current depth in the tree. Default is 0.
        """
        if node is None:
            node = self.tree  # Start from the root
        
        if node["leaf"]:
            print("  " * depth + f"Leaf -> Predict: {node['prediction']}")
        else:
            print("  " * depth + f"Feature {node['feature']} <= {node['threshold']}")
            self.printTree(node["left"], depth + 1)
            print("  " * depth + f"Feature {node['feature']} > {node['threshold']}")
            self.printTree(node["right"], depth + 1)
            
class BaggedTrees:
    """
    Implementation of Bootstrap Aggregating (Bagging) for decision trees.
    
    This class creates an ensemble of decision trees, each trained on a bootstrap
    sample of the dataset. Predictions are made by majority voting. Can be configured
    to work as a Random Forest by enabling the randomForest flag.
    
    Attributes:
        numTrees (int): Number of trees in the ensemble.
        maxDepth (int): Maximum depth for each tree.
        minSamples (int): Minimum samples required to split a node.
        randomForest (bool): If True, builds a Random Forest by using random feature selection.
        trees (list): List of trained decision trees.
        oobVotes (dict): Dictionary tracking out-of-bag predictions.
    """
    
    def __init__(self, numTrees=10, maxDepth=3, minSamples=2, randomForest=False):
        """
        Initialize the BaggedTrees ensemble.
        
        Parameters:
            numTrees (int, optional): Number of trees in the ensemble. Default is 10.
            maxDepth (int, optional): Maximum depth for each tree. Default is 3.
            minSamples (int, optional): Minimum samples required to split a node. Default is 2.
            randomForest (bool, optional): If True, uses random feature selection. Default is False.
        """
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.minSamples = minSamples
        self.randomForest = randomForest
        self.trees = []

    def bootstrapSample(self, X, y):
        """
        Generate a bootstrap sample from the dataset and identify out-of-bag samples.
        
        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (array-like): Target labels.
            
        Returns:
            tuple: (X_bootstrap, y_bootstrap, X_oob, y_oob) where:
                - X_bootstrap, y_bootstrap: Bootstrap sample features and labels
                - X_oob, y_oob: Out-of-bag samples features and labels
        """
        numSamples = len(y)
        indices = np.random.choice(numSamples, numSamples, replace=True)
        oobIndices = [i for i in range(numSamples) if i not in indices]
        return X[indices], y[indices], X[oobIndices], y[oobIndices]
    
    def fit(self, X, y):
        """
        Fit the ensemble model on the provided data.
        
        Trains multiple decision trees on bootstrap samples and tracks out-of-bag predictions.
        
        Parameters:
            X (numpy.ndarray): Feature matrix.
            y (array-like): Target labels.
        """
        self.trees = []
        self.oobVotes = {i:[] for i in range(len(y))}

        for _ in range(self.numTrees):
            Xi, yi, X_oob, y_oob = self.bootstrapSample(X, y)
            tree = BinaryDecisionTree(self.maxDepth, self.minSamples, self.randomForest)
            tree.train(Xi, yi)
            self.trees.append(tree)

            # OOB Predictions
            for i, x in enumerate(X_oob):
                self.oobVotes[i].append(tree.predictSingle(x))

    def predict(self, X):
        """
        Predict class labels for multiple instances using majority voting.
        
        Parameters:
            X (numpy.ndarray): Feature matrix where each row is an instance.
            
        Returns:
            numpy.ndarray: Array of predicted class labels.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Shape (numTrees, numSamples)
        majority_votes = [max(set(preds), key=list(preds).count) for preds in predictions.T]
        return np.array(majority_votes)
        
    def oobError(self, y):
        """
        Compute Out-of-Bag error rate.
        
        This is an unbiased estimate of the generalization error, calculated using
        samples that were not used for training individual trees.
        
        Parameters:
            y (array-like): True target labels.
            
        Returns:
            float: Out-of-Bag error rate (proportion of incorrectly classified samples).
        """
        errors = 0
        total = 0
        for i, votes in self.oobVotes.items():
            if votes:  # Ensure there are votes for the sample
                total += 1
                if max(set(votes), key=votes.count) != y[i]:
                    errors += 1
        return errors / total if total > 0 else 0

    def printTrees(self):
        """
        Print the structure of all trees in the ensemble.
        """
        for i in range(len(self.trees)):
            print(f"\nTree {i}")
            self.trees[i].printTree()
            
class DecisionStump:
    """
    One height trees that store which feature-threshold split is best,
    which side is to be classified as which class and its weight in the final model
    """
    def __init__(self):
        self.featureIndex = None
        self.threshold = None
        self.polarity = 1  # Determines the direction of the inequality
        self.beta = None  # Classifier weight

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.featureIndex] < self.threshold] = -1
        else:
            predictions[X[:, self.featureIndex] >= self.threshold] = -1

        return predictions

    def __str__(self):
        return (f"DecisionStump(featureIndex={self.featureIndex}, "
                f"threshold={self.threshold:.4f}, "
                f"lessThanClass={self.polarity}, "
                f"beta={self.beta:.4f})")

class AdaBoost:
    """
    AdaBoost using decision stumps as weak learners.

    n_estimators: Number of weak learners.
    uniform_cuts: Number of thresholds to try per feature.
    verbose: Print training progress.
    """
    def __init__(self, n_estimators=200, uniform_cuts=3, verbose=True):
        self.n_estimators = n_estimators
        self.uniform_cuts = uniform_cuts
        self.verbose = verbose
        self.stumps = []
        self.train_errors = []
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        n_samples, n_features = X_train.shape
        weights = np.full(n_samples, 1 / n_samples)

        for t in range(self.n_estimators):
            # if self.verbose:
            #     print(f"\nUsing Weights: {weights}")

            stump = DecisionStump()
            min_error = float('inf')

            # Search for best feature and threshold
            for featureIndex in range(n_features):
                feature_values = X_train[:, featureIndex]
                thresholds = np.linspace(np.min(feature_values),
                                       np.max(feature_values),
                                       self.uniform_cuts + 2)[1:-1]

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[feature_values < threshold] = -1
                        else:
                            predictions[feature_values >= threshold] = -1

                        misclassified = weights * (predictions != y_train)
                        error = np.sum(misclassified)

                        if error > 0.5:
                            error = 1 - error
                            polarity = -polarity

                        if error < min_error:
                            min_error = error
                            stump.featureIndex = featureIndex
                            stump.threshold = threshold
                            stump.polarity = polarity

            # Prevent numerical instability
            min_error = np.clip(min_error, 1e-10, 1 - 1e-10)

            # Calculate classifier weight
            stump.beta = 0.5 * np.log((1 - min_error) / min_error)

            # Update weights
            predictions = stump.predict(X_train)
            weights *= np.exp(-stump.beta * y_train * predictions)
            weights /= np.sum(weights)

            self.stumps.append(stump)

            # Store metrics
            self._store_metrics(X_train, y_train, X_val, y_val, X_test, y_test)

            # Handle output based on verbosity
            if self.verbose:
                print(stump)
                print(f"Min Error: {min_error} Misclassified: {np.sum(predictions != y_train)}, "
                      f"Updating as w_j*{(1 - min_error)/min_error:.1f}")
                print(f"Iteration {t + 1}: Train error: {self.train_errors[-1]:.4f}")
            elif (t + 1) % 10 == 0 or t == 0 or t == self.n_estimators - 1:
                print(f"Round {t + 1}/{self.n_estimators}, Train Error: {self.train_errors[-1]:.4f}")

    def _store_metrics(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_pred = self.predict(X_train)
        self.train_errors.append(np.mean(train_pred != y_train))

        for X, y, loss_list in [
            (X_train, y_train, self.train_losses),
            (X_val, y_val, self.val_losses),
            (X_test, y_test, self.test_losses)
        ]:
            margin = -y * self.decision_function(X)
            loss = np.mean(np.exp(np.clip(margin, -500, 500)))
            loss_list.append(loss)

    def decision_function(self, X):
        return sum(stump.beta * stump.predict(X) for stump in self.stumps)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def __str__(self):
        s = f"AdaBoost(numClassifiers={self.n_estimators})\n"
        for i, clf in enumerate(self.stumps):
            s += f"  Round {i + 1}: {clf}\n"
        return s

def gini(y):
    """
    Calculate the Gini impurity for a set of labels.
    
    The Gini impurity measures how often a randomly chosen element would be incorrectly
    labeled if it was randomly labeled according to the distribution of labels in the set.
    
    Parameters:
        y (array-like): Array of class labels.
        
    Returns:
        float: Gini impurity value. Lower values indicate more homogeneous sets.
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts/counts.sum()
    return 1 - np.sum(probabilities**2)

def findBestSplit(X, y):
    """
    Find the best feature and threshold to split the data based on Gini impurity.
    
    This function evaluates all possible features and thresholds to determine
    which split minimizes the weighted average of Gini impurities in the resulting partitions.
    
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (d, n) where d is the number of samples 
                           and n is the number of features.
        y (array-like): Target labels array.
        
    Returns:
        tuple: (bestFeature, bestThreshold) where:
            - bestFeature (int): Index of the feature that gives the best split.
            - bestThreshold (float): Threshold value for the selected feature.
            Returns (None, None) if no valid split is found.
    """
    bestGini = float('inf')
    bestFeature = None
    bestThreshold = None
    d, n = X.shape

    for feature in range(n):
        unique_values = np.unique(X[:, feature])  # Unique sorted values
        if len(unique_values) < 2:
            continue  # Skip if only one unique value (no split possible)
        # Compute midpoints of consecutive unique values
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    
        for threshold in midpoints:
            leftSplit = X[:, feature] <= threshold
            rightSplit = X[:, feature] > threshold

            if(sum(leftSplit)*sum(rightSplit) == 0): #either is 0 (all false), then move on 
                continue
                
            leftGini = gini(y[leftSplit])
            rightGini = gini(y[rightSplit])
            totalGini = (sum(leftSplit) * leftGini + sum(rightSplit) * rightGini) / len(y)
                       
            if totalGini < bestGini:
                bestGini = totalGini
                bestFeature = feature
                bestThreshold = threshold
    return bestFeature, bestThreshold

def findBestSplitRF(X, y, numFeatures=2):
    """
    Find the best split for Random Forest by evaluating only a subset of features.
    
    Similar to findBestSplit but considers only a randomly selected subset of features,
    which is a key characteristic of the Random Forest algorithm.
    
    Parameters:
        X (numpy.ndarray): Feature matrix of shape (d, n) where d is the number of samples 
                           and n is the number of features.
        y (array-like): Target labels array.
        numFeatures (int, optional): Number of features to consider for the split. Default is 2.
        
    Returns:
        tuple: (bestFeature, bestThreshold) where:
            - bestFeature (int): Index of the feature that gives the best split.
            - bestThreshold (float): Threshold value for the selected feature.
            Returns (None, None) if no valid split is found.
    """
    bestGini = float('inf')
    bestFeature = None
    bestThreshold = None

    # Randomly select a subset of features
    featureIndices = np.random.choice(X.shape[1], numFeatures, replace=False)

    for feature in featureIndices:
        unique_values = np.unique(X[:, feature])  # Unique sorted values
        if len(unique_values) < 2:
            continue  # Skip if only one unique value (no split possible)

        # Compute midpoints of consecutive unique values
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in midpoints:
            leftSplit = X[:, feature] <= threshold
            rightSplit = X[:, feature] > threshold

            if sum(leftSplit) == 0 or sum(rightSplit) == 0:  # Avoid empty splits
                continue

            leftGini = gini(y[leftSplit])
            rightGini = gini(y[rightSplit])
            totalGini = (sum(leftSplit) * leftGini + sum(rightSplit) * rightGini) / len(y)

            if totalGini < bestGini:
                bestGini = totalGini
                bestFeature = feature
                bestThreshold = threshold

    return bestFeature, bestThreshold

def foldSplit(data, k=5):
    """
    Splits data into k folds for cross-validation.
    
    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The dataset to split.
        k (int, optional): Number of folds. Default is 5.
        
    Returns:
        list: A list of k arrays containing indices for each fold.
    """
    indices = np.arange(len(data)) # Generate array with evenly spaced values
    np.random.shuffle(indices)
    return np.array_split(indices, k)

def polyFeatures(x, degree):
    """
    Creates polynomial features from a single feature.
    
    Transforms a single feature into a matrix with polynomial terms up to the specified degree.
    
    Parameters:
        x (numpy.ndarray): Input feature vector.
        degree (int): Maximum polynomial degree.
        
    Returns:
        numpy.ndarray: Matrix with polynomial features where each column represents
                       a different power of x from 0 to degree.
    """
    return np.column_stack([x**d for d in range(degree + 1)])

def crossValError(data, degree, k=5):
    """
    Performs k-fold cross-validation for polynomial regression and returns average MSE.
    
    This function splits the data into k folds, trains a polynomial regression model 
    on k-1 folds and tests on the remaining fold, repeating for each fold.
    
    Parameters:
        data (pandas.DataFrame): Dataset containing 'x' and 'y' columns.
        degree (int): Polynomial degree to use in the regression.
        k (int, optional): Number of folds for cross-validation. Default is 5.
        
    Returns:
        float: Average Mean Squared Error across all folds.
    """
    folds = foldSplit(data, k)
    errors = []
        
    for i in range(k):
        # Create train-test split
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        train_x, train_y = data.iloc[train_idx]['x'].values, data.iloc[train_idx]['y'].values
        test_x, test_y = data.iloc[test_idx]['x'].values, data.iloc[test_idx]['y'].values
        # Generate polynomial features
        X_train = polyFeatures(train_x, degree)
        X_test = polyFeatures(test_x, degree)

        W = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ train_y # Compute weights

        # Predict on test data
        y_pred = X_test @ W

        # Compute Mean Squared Error (MSE)
        mse = np.mean((test_y - y_pred) ** 2)
        errors.append(mse)

    return np.mean(errors)

def plot_results(adaboost):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(adaboost.train_losses, label='Train Loss')
    plt.plot(adaboost.test_losses, label='Test Loss')
    plt.plot(adaboost.val_losses, label='Validation Loss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Exponential Loss')
    plt.legend()
    plt.title('Loss vs Boosting Rounds')

    # Error plot
    plt.subplot(1, 2, 2)
    plt.plot(adaboost.train_errors, label='Train Error')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.title('Training Error vs Boosting Rounds')

    plt.tight_layout()
    plt.show()

def plot_stump_weights(adaboost):
    """Plot the weights (beta) assigned to each stump across rounds"""
    rounds = np.arange(1, len(adaboost.stumps)+1)
    betas = [stump.beta for stump in adaboost.stumps]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, betas, marker='o', linestyle='-', color='b')
    plt.xlabel('Boosting Round')
    plt.ylabel('Stump Weight (β)')
    plt.title('Classifier Weight vs Boosting Round')
    plt.grid(True)

    # Highlight every 10th round for readability
    for r in range(0, len(rounds), 10):
        plt.annotate(f"{betas[r]:.3f}",
                    (rounds[r], betas[r]),
                    textcoords="offset points",
                    xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.show()
