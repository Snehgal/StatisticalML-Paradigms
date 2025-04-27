import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def numpy_split(x, y, test_size=0.2, random_state=None):
    """
    Split dataset into train and test sets.
    
    Parameters:
        x (np.ndarray): Features (n_samples, n_features).
        y (np.ndarray): Labels (n_samples,).
        test_size (float): Fraction of data to reserve for testing.
        random_state (int, optional): Random seed for reproducibility.
    
    Returns:
        x_train, x_test, y_train, y_test (np.ndarray): Splits of input data.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(x)
    n_test = int(n_samples * test_size)

    # Create shuffled indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Split indices
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    # Split data
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return x_train, x_test, y_train, y_test
        
def accuracy(xTest, labels, predict):
    """
    Compute classification accuracy.

    Parameters:
        xTest (np.ndarray): Test features of shape (n_features, n_samples).
        labels (np.ndarray): True labels.
        predict (function): Function that predicts label given a sample.

    Returns:
        float: Accuracy (correct predictions / total predictions).
    """
    correct = 0
    for i in range(xTest.shape[1]): 
        x = xTest[:, i]  # single test sample
        a = predict(x)
        if a == labels[i]:
            correct += 1
    return correct / xTest.shape[1]

def pca(X, preservedVariance=None, pcaComponents=None):
    """
    Perform Principal Component Analysis (PCA) on the data.

    Parameters:
        X (np.ndarray): Data matrix of shape (features, samples).
        preservedVariance (float, optional): Desired variance retention (e.g., 0.95 or 95 for 95%).
        pcaComponents (int, optional): Number of PCA components to reduce to.
    
    Returns:
        X_reduced (np.ndarray): Reduced data matrix (p components x samples).
        U_p (np.ndarray): Top eigenvectors (features x p components).
        p (int): Number of components selected.
    """
    u = np.mean(X, axis=1, keepdims=True)
    X_centered = X - u

    cov = np.dot(X_centered, X_centered.T) / (X.shape[1] - 1)

    eigenVals, eigenVecs = np.linalg.eig(cov)
    sortIndices = np.argsort(eigenVals)[::-1]
    eigenVals = eigenVals[sortIndices]
    eigenVecs = eigenVecs[:, sortIndices]

    if preservedVariance is not None:
        if preservedVariance > 1:
            preservedVariance /= 100
        total_variance = np.sum(eigenVals)
        variance_sum = 0
        p = 0
        while variance_sum <= preservedVariance and p < len(eigenVals):
            variance_sum += eigenVals[p] / total_variance
            p += 1
        print(f"Dimensions that data is reduced to for retaining {preservedVariance*100:.2f}% variance: {p}")
    elif pcaComponents is not None:
        p = pcaComponents
        print(f"Reducing to {p} PCA components (user-specified).")
    else:
        raise ValueError("Either preservedVariance or pcaComponents must be provided.")

    U_p = eigenVecs[:, :p]
    X_reduced = np.dot(U_p.T, X_centered)

    return X_reduced, U_p, p

def FDA(train_sets, delta=1e-3, num_components=2, graph=False):
    """
    Perform Fisher Discriminant Analysis (FDA) on training sets.

    Parameters:
        train_sets (list of np.ndarray): List of datasets, each (features, samples) for a class.
        delta (float): Regularization parameter for stability.
        num_components (int): Number of FDA components to retain.
        graph (bool): Whether to plot the projected data.

    Returns:
        W (np.ndarray): Projection matrix (features x num_components).
        X_projected (np.ndarray): Projected dataset (num_components x total_samples).
    """
    num_classes = len(train_sets)
    feature_dim = train_sets[0].shape[0]

    # Step 1: Compute class means
    class_means = [np.mean(X, axis=1, keepdims=True) for X in train_sets]

    global_mean = np.mean(np.hstack(class_means), axis=1, keepdims=True)

    # Step 2: Compute within-class scatter matrix S_w
    S_w = np.zeros((feature_dim, feature_dim))
    for X, u in zip(train_sets, class_means):
        for i in range(X.shape[1]):
            diff = (X[:, i].reshape(-1, 1) - u)
            S_w += diff @ diff.T

    # Step 3: Compute between-class scatter matrix S_b
    S_b = np.zeros((feature_dim, feature_dim))
    for u in class_means:
        diff = u - global_mean
        S_b += diff @ diff.T

    # Step 4: Solve generalized eigenvalue problem
    S_w_reg = S_w + delta * np.eye(feature_dim)
    eigvals, eigvecs = eigh(S_b, S_w_reg, eigvals_only=False)

    # Step 5: Sort eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]
    
    # Step 6: Select top eigenvectors
    W = eigvecs[:, :num_components]

    # Step 7: Project all data
    X_all = np.hstack(train_sets)
    X_projected = W.T @ X_all

    # Step 8: Optional visualization
    if graph:
        labels = np.hstack([np.full(X.shape[1], idx) for idx, X in enumerate(train_sets)])

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_projected[0, :], X_projected[1, :], c=labels, cmap='viridis', marker='o')
        plt.colorbar(scatter, label='Class Label')
        plt.title('2D FDA Projection')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.show()

    return W, X_projected
