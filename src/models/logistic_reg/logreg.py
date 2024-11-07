from numpy import dot, array
from math import exp, log, ceil
import random

EPOCHS = 3
EPSILON = 1e-9
LEARNING_RATE = 0.1


def sigmoid(z) -> float:
    """Classic sigmoid activation function, aka logistic function.
    activation functions are used to introduce non-linearity into the model.
    Sigmoid is used to squash the output of the linear function to a range of
    0 to 1.
    z: the output of the linear function
    (dot product of the weights and the input features)"""
    return 1 / (1 + exp(-z))


def loss_func(y, y_hat, epsilon=EPSILON) -> float:
    """Evaluate the loss function for a given prediction
    y: target label
    y_hat: predicted label
    epsilon: small value to prevent log(0)"""
    # binary cross-entropy loss
    return -y * log(y_hat + epsilon) - (1 - y) * log(1 - y_hat + epsilon)


class LogisticRegression:
    def __init__(self, theta=None, bias=0):
        self.theta = theta if theta else [0.1, 0.2]
        self.bias = bias

    def forward(self, X=None, bias=0) -> float:
        """Forward pass of the model without activation function, aka linear.
        returns the dot product of the weights and the input features
        X: input features
        bias: bias term"""
        return dot(self.theta.T, X) + bias

    def backward(self, y, y_hat, m=1) -> float:
        """Backward pass of the model, aka backpropagation.
        We use the error term and multiply it by the weights to get the
        gradient.
        y: target label
        y_hat: predicted label
        m: number of samples in the dataset
        """
        # calculate the loss (aka error; aka cost)
        error_term = y_hat - y
        print(f"Error term: {error_term}")
        # calculate the gradient of the loss function
        # we can think of this like the forward pass in reverse
        # we are looking to see how the loss changes with respect to the weights
        # aka how the changes in weights contribute to the loss
        grads = (1 / m) * dot(self.theta, error_term.T)
        return grads

    def stochastic_gradient_descent(
            self, X, Y, epochs=EPOCHS, learning_rate=LEARNING_RATE):
        """
        Stochastic Gradient Descent (SGD) is an optimization algorithm used to
        minimize some function by iteratively moving in the direction of
        steepest descent as defined by the negative of the gradient.
        X: input features
        Y: target labels
        epochs: number of iterations
        learning_rate: step size"""
        for epoch in range(EPOCHS):
            for i, x in enumerate(X):
                # forward pass
                logit = self.forward(x)
                # activation function
                y_hat = sigmoid(logit)
                # calculate loss
                # the loss shows how well the model is doing but
                # is not used to update the weights directly
                loss = loss_func(Y[i], y_hat)
                # backward pass to calculate the gradients using the error term
                grads = self.backward(y_hat, Y[i], m=len(X))
                # update weights
                self.theta -= learning_rate * grads
                print(f"Epoch: {epoch}, Loss: {loss}, Error: {loss}")
        return self.theta

    def train(self, X, Y, seed=42):
        """Train the model using the input features and the target labels
        X: input features
        Y: target labels
        """
        # convert to numpy array
        X = array(X)
        Y = array(Y)
        # ensure n randomized weights for each feature
        features_n = X.shape[1]
        self.theta = array([random.random() for _ in range(features_n)])
        # train the model
        self.stochastic_gradient_descent(X, Y)
        return self.theta

    def predict(self, x):
        """Get the output (y-hat) of the model for a given input"""
        logit = self.forward(x)
        prob = sigmoid(logit)
        return int(prob > 0.5), prob


def main() -> int:
    # Data
    X = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
    Y = [0, 1, 1, 1]
    X_test = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]]
    y_test = [0, 1, 1, 1]
    model = LogisticRegression()
    theta = model.train(X, Y)
    # Test
    for i, x in enumerate(X_test):
        y_hat, p = model.predict(x)
        print(f"Prediction: {y_hat}, Ground Truth: {y_test[i]}, prob: {p}")
    print(f"Final weights: {theta}")
    return 0


if '__main__' == __name__:
    main()
