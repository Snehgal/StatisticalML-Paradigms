from .DataProcessing import pca,numpy_split,FDA
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error

class DecisionStump:
    """
    Represents a one-level decision tree (decision stump) for regression.

    Attributes:
        featureIndex (int or None): Index of the feature to split on (not used in 1D input here).
        threshold (float or None): Threshold value to split the data.
        left_value (float or None): Prediction value for samples less than or equal to the threshold.
        right_value (float or None): Prediction value for samples greater than the threshold.
    """

    def __init__(self):
        """
        Initializes the DecisionStump with empty parameters.
        """
        self.featureIndex = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals, n_cuts=20):
        """
        Trains the DecisionStump by finding the best threshold to fit the residuals.

        Args:
            X (np.ndarray): Input feature array of shape (n_samples,).
            residuals (np.ndarray): Residual target values to fit.
            n_cuts (int): Number of candidate threshold cuts to consider.

        Returns:
            DecisionStump: The fitted DecisionStump instance.
        """
        best_loss = float('inf')
        X = X.flatten()
        thresholds = np.linspace(0, 1, n_cuts + 2)[1:-1]

        for threshold in thresholds:
            left_mask = []
            right_mask = []
            for i in X:
                if i <= threshold:
                    left_mask.append(True)
                    right_mask.append(False)
                else:
                    left_mask.append(False)
                    right_mask.append(True)

            left_value = np.mean(residuals[left_mask]) if np.any(left_mask) else 0
            right_value = np.mean(residuals[right_mask]) if np.any(right_mask) else 0

            predictions = np.where(left_mask, left_value, right_value)
            loss = np.mean((residuals - predictions) ** 2)

            if loss < best_loss:
                best_loss = loss
                self.threshold = threshold
                self.left_value = left_value
                self.right_value = right_value

        return self

    def predict(self, X):
        """
        Predicts output values for given input X using the fitted threshold.

        Args:
            X (np.ndarray): Input feature array of shape (n_samples,).

        Returns:
            np.ndarray: Predicted output array of shape (n_samples,).
        """
        X = X.flatten()
        return np.where(X <= self.threshold, self.left_value, self.right_value)

    def __str__(self):
        """
        Returns a string representation of the DecisionStump.
        """
        return (f"DecisionStump(featureIndex={self.featureIndex}, threshold={self.threshold:.4f}, "
                f"left_value={self.left_value:.4f}, right_value={self.right_value:.4f})")

class GradientBoosting:
    """
    Implements a simple Gradient Boosting Machine (GBM) using Decision Stumps as weak learners.

    Attributes:
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Learning rate (shrinkage factor).
        loss (str): Loss function to use ('squared' or 'absolute').
        stumps (list): List of fitted DecisionStump instances.
        train_loss_history (list): Training loss history after each iteration.
        uniform_cuts (int): Number of threshold cuts considered when fitting stumps.
    """

    def __init__(self, n_estimators=100, learning_rate=0.01, loss='squared', uniform_cuts=20):
        """
        Initializes the GradientBoosting instance.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.stumps = []
        self.train_loss_history = []
        self.uniform_cuts = uniform_cuts
        self.y_train = None
    def _negative_gradient(self, y, y_pred):
        """
        Computes the negative gradient (residuals) based on the loss function.

        Args:
            y (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: Residuals.
        """
        if self.loss == 'squared':
            return y - y_pred
        elif self.loss == 'absolute':
            return np.sign(y - y_pred)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        """
        Trains the Gradient Boosting model on the input data.

        Args:
            X (np.ndarray): Input feature array of shape (n_samples,).
            y (np.ndarray): Target array of shape (n_samples,).

        Returns:
            GradientBoosting: The fitted model instance.
        """
        self.y_train = y
        initial_pred = np.mean(y)
        y_pred = np.full_like(y, initial_pred)

        for _ in range(self.n_estimators):
            residuals = self._negative_gradient(y, y_pred)
            stump = DecisionStump().fit(X, residuals, n_cuts=self.uniform_cuts)
            update = stump.predict(X)
            y_pred += self.learning_rate * update

            self.stumps.append(stump)
            current_loss = np.mean((y - y_pred) ** 2) if self.loss == 'squared' else np.mean(np.abs(y - y_pred))
            self.train_loss_history.append(current_loss)

        return self

    def predict(self, X):
        """
        Predicts output values for input X using the trained model.

        Args:
            X (np.ndarray): Input feature array.

        Returns:
            np.ndarray: Predicted output array.
        """
        if self.loss == 'squared':
            y_pred = np.full(X.shape[0], np.mean(self.y_train))
        else:
            y_pred = np.full(X.shape[0], np.median(self.y_train))

        for stump in self.stumps:
            y_pred += self.learning_rate * stump.predict(X)

        return y_pred

    def __str__(self):
        """
        Returns a string representation of the GradientBoosting model.
        """
        stumps_str = "\n".join([f"  {i+1}: {str(stump)}" for i, stump in enumerate(self.stumps)])
        return (f"GradientBoosting(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, loss='{self.loss}')\n"
                f"Stumps:\n{stumps_str}")

class SimpleNeuralNetwork:
    """
    Implements a simple fully connected neural network with one hidden layer.

    Attributes:
        W1 (np.ndarray): Weight matrix for input to hidden layer.
        W2 (np.ndarray): Weight matrix for hidden to output layer.
        b1 (np.ndarray): Bias vector for hidden layer.
        b2 (np.ndarray): Bias vector for output layer.
    """

    def __init__(self):
        """
        Initializes the neural network with random weights and biases.
        """
        self.W1 = np.random.randn(2, 1)
        self.W2 = np.random.randn(1, 1)
        self.b1 = np.random.randn(1, 1)
        self.b2 = np.random.randn(1, 1)

    def sigmoid(self, x):
        """
        Computes the sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid activation output.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        Performs forward propagation through the network.

        Args:
            X (np.ndarray): Input data of shape (n_samples, 2).

        Returns:
            np.ndarray: Output predictions.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def compute_loss(self, y, y_hat):
        """
        Computes mean squared error (MSE) loss.

        Args:
            y (np.ndarray): True target values.
            y_hat (np.ndarray): Predicted values.

        Returns:
            float: Computed MSE loss.
        """
        return np.mean(0.5 * (y - y_hat) ** 2)

    def backward(self, X, y, y_hat, learning_rate, method):
        """
        Performs backward propagation and updates weights.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            y_hat (np.ndarray): Predicted outputs.
            learning_rate (float): Learning rate for weight updates.
            method (str): Training method - "gradientDescent" (batch) or "stochastic" (sample-wise).
        """
        y = y.reshape(-1, 1)
        if method == "gradientDescent":
            m = X.shape[0]
            dW2 = (1/m) * np.dot(self.a1.T, y_hat - y)
            db2 = (1/m) * np.sum(y_hat - y, axis=0, keepdims=True)
            dz1 = np.dot(y_hat - y, self.W2.T) * self.a1 * (1 - self.a1)
            dW1 = (1/m) * np.dot(X.T, dz1)
            db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        elif method == "stochastic":
            dW2 = np.dot(self.a1.T, y_hat - y)
            db2 = y_hat - y
            dz1 = np.dot(y_hat - y, self.W2.T) * self.a1 * (1 - self.a1)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, learning_rate=0.1, epochs=1000, method="gradientDescent"):
        """
        Trains the neural network using either batch or stochastic gradient descent.

        Args:
            X (np.ndarray): Training input data.
            y (np.ndarray): True labels.
            learning_rate (float): Learning rate for updates.
            epochs (int): Number of training iterations.
            method (str): "gradientDescent" for batch or "stochastic" for sample-wise updates.
        """
        print(f"Training Parameters\nMethod:{method} Learning Rate:{learning_rate} Epochs:{epochs}")
        if method == "gradientDescent":
            for epoch in range(epochs):
                y_hat = self.forward(X)
                loss = self.compute_loss(y, y_hat)
                self.backward(X, y, y_hat, learning_rate, method)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

        elif method == "stochastic":
            for epoch in range(epochs):
                for i in range(X.shape[0]):
                    x_sample = X[i:i+1, :]
                    y_sample = y[i:i+1, :] if len(y.shape) > 1 else y[i:i+1]
                    y_hat = self.forward(x_sample)
                    loss = self.compute_loss(y_sample, y_hat)
                    self.backward(x_sample, y_sample, y_hat, learning_rate, method)

                if epoch % 100 == 0:
                    full_y_hat = self.forward(X)
                    full_loss = self.compute_loss(y, full_y_hat)
                    print(f"Epoch {epoch}, Loss: {full_loss:.4f}")
            print(f"Epoch {epoch}, Loss: {full_loss:.4f}")

        else:
            raise ValueError("Invalid method. Choose 'gradientDescent' or 'stochastic'")

    def __str__(self):
        """
        Returns a string representation of the neural network's parameters.
        """
        return (f"Weights and biases:\n"
                f"W1:\n{self.W1}\n"
                f"b1:\n{self.b1}\n"
                f"W2:\n{self.W2}\n"
                f"b2:\n{self.b2}\n")

def plot_predictions_vs_truth(model, X, y, title, ax):
    """
    Plots model predictions against true values at selected boosting iterations.

    Parameters:
    - model: Trained GradientBoosting model.
    - X: Input features (numpy array).
    - y: True target values (numpy array).
    - title: Title for the plot (string).
    - ax: Matplotlib Axes object to plot on.

    Returns:
    - None
    """
    x_sorted = np.sort(X.flatten())
    idx = np.argsort(X.flatten())
    y_sorted = y[idx]

    predictions = np.zeros_like(y_sorted)
    if model.loss == 'squared':
        predictions += np.mean(y_train)
    else:
        predictions += np.median(y_train)

    for i, stump in enumerate(model.stumps):
        predictions += learning_rate * stump.predict(x_sorted)
        if i in [0, 9, 49, 99, 199, 499, 999]:  # Plot at different iterations
            ax.plot(x_sorted, predictions, label=f'Iter {i+1}', alpha=0.7)

    ax.scatter(x_sorted, y_sorted, color='black', s=10, alpha=0.5, label='True values')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

def plot_training_loss(models, model_names):
    """
    Plots training loss curves for multiple models over boosting iterations.

    Parameters:
    - models: List of trained GradientBoosting models.
    - model_names: List of names corresponding to models.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    for model, name in zip(models, model_names):
        plt.plot(range(1, n_estimators+1), model.train_loss_history, label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_accuracy(model, X_test, y_test):
    """
    Calculates R-squared score between model predictions and ground truth.

    Parameters:
    - model: Trained GradientBoosting model.
    - X_test: Test input features (numpy array).
    - y_test: True target values (numpy array).

    Returns:
    - r_squared: R² score (float)
    """
    predictions = model.predict(X_test)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def gaussian(x, u, cov):
    """
    Computes log probability density of multivariate Gaussian distribution.

    Parameters:
    - x: Input sample (numpy array).
    - u: Mean vector (numpy array).
    - cov: Covariance matrix (numpy array).

    Returns:
    - log_likelihood: Log probability of sample under Gaussian (float)
    """
    d = x.shape[0]
    u = u.reshape(d)
    a = (x - u).reshape(d, 1)
    delta = 1e-6 
    cov = cov + delta * np.eye(cov.shape[0])  # Regularization
    det = np.linalg.det(cov)
    if det == 0:
        det = np.e ** (-15)
    log_det = np.log(det)
    mahalanobis = np.dot(np.dot(a.T, np.linalg.inv(cov)), a)
    return -0.5 * (mahalanobis + log_det)

def predictMLE(x, u0, cov0, u1, cov1, u2, cov2):
    """
    Predict class label for a sample using Maximum Likelihood Estimation (MLE).

    Parameters:
    - x: Input sample (numpy array).
    - u0, u1, u2: Class means (numpy arrays).
    - cov0, cov1, cov2: Class covariance matrices (numpy arrays).

    Returns:
    - predicted_label: Predicted class label (int: 0, 1, or 2)
    """
    Px0 = gaussian(x, u0, cov0)
    Px1 = gaussian(x, u1, cov1)
    Px2 = gaussian(x, u2, cov2)
    if (Px0 > Px1 and Px0 > Px2):
        return 0
    if (Px1 > Px0 and Px1 > Px2):
        return 1
    else:
        return 2

def discriminantLDA(X, us, cov):
    """
    Compute LDA discriminant scores for each class.

    Parameters:
    - X: Input features (numpy array of shape [features, samples]).
    - us: List of class means (list of numpy arrays).
    - cov: Shared covariance matrix (numpy array).

    Returns:
    - scores: Discriminant scores for each class (numpy array)
    """
    d = X.shape[0] 
    delta = 1e-6
    cov = cov + delta * np.eye(d)  # Regularization
    covInv = np.linalg.inv(cov)  
    U = np.concatenate(us, axis=1)
    return (covInv @ U).T @ X - 0.5 * np.sum(U * (covInv @ U), axis=0, keepdims=True).T

def predictLDA(X, us, cov):
    """
    Predict class labels using Linear Discriminant Analysis (LDA).

    Parameters:
    - X: Input features (numpy array of shape [features, samples]).
    - us: List of class means (list of numpy arrays).
    - cov: Shared covariance matrix (numpy array).

    Returns:
    - predicted_labels: Predicted class labels (numpy array)
    """
    discriminants = discriminantLDA(X, us, cov)
    return np.argmax(discriminants, axis=0)

def discriminantQDA(X, us, covs,priors=None):
    """
    Compute QDA discriminant scores for each class.

    Parameters:
    - X: Input features (numpy array of shape [features, samples]).
    - us: List of class means (list of numpy arrays).
    - covs: List of class covariance matrices (list of numpy arrays).
    - priors: List of class prior probabilities (list of floats).

    Returns:
    - scores: Discriminant scores for each class (numpy array)
    """
    num_classes = len(us)
    if priors is None:
        priors = [1/num_classes]*num_classes
    num_samples = X.shape[1]
    scores = np.zeros((num_classes, num_samples))
    
    for i in range(num_classes):
        d = X.shape[0]
        delta = 1e-6
        cov = covs[i] + delta * np.eye(d)
        covInv = np.linalg.inv(cov)
        detCov = np.linalg.det(cov)
        
        mean_diff = X - us[i]
        quad_term = -0.5 * np.sum(mean_diff * (covInv @ mean_diff), axis=0)
        log_det_term = -0.5 * np.log(detCov)
        log_prior = np.log(priors[i]) if priors is not None else 0
        
        scores[i, :] = quad_term + log_det_term + log_prior
    
    return scores

def predictQDA(X, us, covs,priors=None):
    """
    Predict class labels using Quadratic Discriminant Analysis (QDA).

    Parameters:
    - X: Input features (numpy array of shape [features, samples]).
    - us: List of class means (list of numpy arrays).
    - covs: List of class covariance matrices (list of numpy arrays).
    - priors: List of class prior probabilities (list of floats).

    Returns:
    - predicted_labels: Predicted class labels (numpy array)
    """
    num_classes = len(us)
    if priors is None:
        priors = [1/num_classes]*num_classes
    discriminants = discriminantQDA(X, us, covs, priors)
    return np.argmax(discriminants, axis=0)

def PCAthenLDA(presVar, numPCA=0):
    """
    Perform Principal Component Analysis (PCA) followed by Linear Discriminant Analysis (LDA) on the dataset.
    
    Parameters:
    presVar : float
        The preserved variance for PCA, which determines how much of the variance to keep during the PCA transformation.
    
    numPCA : int, optional, default=0
        The number of principal components to retain for PCA. If 0, the number of components is automatically determined by `presVar`.
    
    Returns:
    predictionsTestPCA_LDA : ndarray
        The predicted class labels for the test set after applying PCA and LDA.
    
    predictionsTrainPCA_LDA : ndarray
        The predicted class labels for the training set after applying PCA and LDA.

    """
    Up, p = getP(preservedVariance=presVar, numPCA=numPCA)

    testPCA = Up.T @ testSetStacked
    trainPCA = Up.T @ trainSetStacked
    train0 = Up.T @ trainSetStacked0  
    train1 = Up.T @ trainSetStacked1  
    train2 = Up.T @ trainSetStacked2
    
    PCAu0 = np.sum(train0, axis=1, keepdims=True) / train0.shape[1]
    PCAu1 = np.sum(train1, axis=1, keepdims=True) / train1.shape[1]
    PCAu2 = np.sum(train2, axis=1, keepdims=True) / train2.shape[1]
    PCAcov0 = np.dot((train0 - PCAu0), (train0 - PCAu0).T) / train0.shape[1]
    PCAcov1 = np.dot((train1 - PCAu1), (train1 - PCAu1).T) / train1.shape[1]
    PCAcov2 = np.dot((train2 - PCAu2), (train2 - PCAu2).T) / train2.shape[1]
    PCAcov = (PCAcov0 + PCAcov1 + PCAcov2) / 3.0
    predictionsTestPCA_LDA = predictLDA(testPCA, [PCAu0, PCAu1, PCAu2], PCAcov)
    predictionsTrainPCA_LDA = predictLDA(trainPCA, [PCAu0, PCAu1, PCAu2], PCAcov)
    
    return predictionsTestPCA_LDA, predictionsTrainPCA_LDA
