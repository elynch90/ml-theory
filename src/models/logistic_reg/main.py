from numpy import dot
from math import exp, log, ceil

EPOCHS = 3
EPSILON = 1e-9
LEARNING_RATE = 0.1


def sigmoid(z) -> float:
    """"""
    return 1 / (1 + exp(-z))


def forward(x=None, theta=None, bias=0) -> float:
    """"""
    return dot(x, theta) + bias


def loss_func(y, y_hat, epsilon=EPSILON) -> float:
    """"""
    return -y * log(y_hat + epsilon) - (1 - y) * log(1 - y_hat + epsilon)


def stochastic_gradient_descent(
        X, Y, theta, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """"""
    for epoch in range(EPOCHS):
        for i, x in enumerate(X):
            # forward pass
            logit = forward(x, theta)
            # activation function
            y_hat = sigmoid(logit)
            # calculate loss
            loss = loss_func(Y[i], y_hat)
            # calculate error (derivative of the loss function)
            error = y_hat - Y[i]
            print(f"Epoch: {epoch}, Loss: {loss}, Error: {error}")
            # backpropagation: update weights
            for j in range(len(theta)):
                # subtracting the derivative of the loss function
                # the gradients tell us which direction will increase the loss
                # so we subtract the gradient to decrease the loss instead
                # like opposite George
                theta[j] -= learning_rate * error * x[j]
    return theta


def predict(x, theta):
    """"""
    logit = forward(x, theta)
    return ceil(sigmoid(logit))


def main() -> int:
    # Data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 1]
    X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_test = [0, 1, 1, 1]
    # initialize weights
    theta = [0.1, 0.2]
    # Train model aka optimize weights
    theta = stochastic_gradient_descent(X, Y, theta)
    # Test
    for i, x in enumerate(X_test):
        logit = forward(x, theta)
        y_hat = ceil(sigmoid(logit))
        print(f"Prediction: {y_hat}, Ground Truth: {y_test[i]}")
    print(f"Final weights: {theta}")
    return 0


if '__main__' == __name__:
    main()
