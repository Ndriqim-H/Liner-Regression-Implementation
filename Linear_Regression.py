import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000):
        if(len(X.shape) == 1):
            num_samples = len(X)
            num_features = 1
        else:
            num_samples, num_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(num_iterations):
            # Calculate predicted values
            if(num_features == 1):
                y_pred = self.weights * X + self.bias
            else:
                y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    #Return the line of best fit
    def line(self, x):
        return self.weights*x + self.bias
    
    #J = (1/2m) * sum(predictions - y)^2
    def J(x,y,theta):
        m = len(y)
        predictions = x.dot(theta)
        cost = (1/2*m) * np.sum(np.square(predictions-y))
        return cost
