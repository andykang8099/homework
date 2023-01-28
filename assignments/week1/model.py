import numpy as np

class LinearRegression:
    """
    A linear regression model that uses the closed-form solution to derive the parameters
    """
    w: np.ndarray
    b: float

    def __init__(self):
        """
        initialize the parameters weight and bias 

        Arguments: 
            None
        
        Returns:
            None

        """
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None :
        """
        This is the method for fitting the data using the analytical solution result.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input data.

        Returns:
            None

        """
        X_append = np.hstack((X, np.ones((len(X), 1))))
        params = []
        # check the singularity of matrix
        if np.linalg.det(X_append.T @ X_append) != 0:
            params = np.linalg.inv(X_append.T @ X_append) @ X_append.T @ y
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")
        self.w = params[:-1]
        self.b = params[-1]


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        y_pred = X @ (self.w).T + self.b
        return y_pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        This part is the data fitting and model training part with the gradient descent method. For 
        the analytical solution of gradient descent, I use the code from the following article as reference.
        https://towardsdatascience.com/implementing-linear-regression-with-gradient-descent-from-scratch-f6d088ec1219

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input data.
            lr: float
            epochs: int

        Returns:
            None

        """
        # Feature scaling
        X_scaled = X
        for col in range(X.shape[1]):
            X_scaled[:, col] = (X[:, col] - np.mean(X[:, col])) / np.std(X[:, col])

        # assign the initial value of parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0
        errors = []
        # Steps for gradient descent
        for i in range(epochs):
            y_pred = self.predict(X_scaled)
            # calculate the mean-squared error
            error = np.sum((y_pred - y) ** 2) / X_scaled.shape[0]
            errors.append(error)
            # compute the partial derivative of parameters
            dw = np.dot((y_pred - y), X_scaled) / X_scaled.shape[0]
            db = np.sum(y_pred - y) / X_scaled.shape[0]
            self.w = self.w - lr * dw
            self.b = self.b - lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # Feature scaling
        X_scaled = X
        for col in range(X.shape[1]):
            X_scaled[:, col] = (X[:, col] - np.mean(X[:, col])) / np.std(X[:, col])

        y_pred = X_scaled @ (self.w).T + self.b
        return y_pred
