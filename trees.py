import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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