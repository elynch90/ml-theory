from numpy import dot
from math import exp, log, ceil

EPOCHS = 3
EPSILON = 1e-9
LEARNING_RATE = 0.1


def sigmoid(z):
    return 1 / (1 + exp(-z))


def forward(x=None, theta=None, bias=0):
    return dot(x, theta) + bias


def loss_func(y, y_hat):
    return -y * log(y_hat + EPSILON) - (1 - y) * log(1 - y_hat + EPSILON)


def main() -> int:
    # Data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 1]
    X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_test = [0, 1, 1, 1]
    # initialize weights
    theta = [0.1, 0.2]
    for epoch in range(EPOCHS):
        for i, x in enumerate(X):
            # forward pass
            logit = forward(x, theta)
            # activation function
            y_hat = sigmoid(logit)
            # calculate loss
            loss = loss_func(y[i], y_hat)
            # calculate error (derivative of the loss function)
            error = y_hat - y[i]
            print(f"Epoch: {epoch}, Loss: {loss}, Error: {error}")
            # backpropagation: update weights
            for j in range(len(theta)):
                # subtracting the derivative of the loss function
                # the gradients tell us which direction will increase the loss
                # so we subtract the gradient to decrease the loss instead
                # like opposite George
                theta[j] -= LEARNING_RATE * error * x[j]
    # Test
    for i, x in enumerate(X_test):
        logit = forward(x, theta)
        y_hat = ceil(sigmoid(logit))
        print(f"Prediction: {y_hat}, Ground Truth: {y_test[i]}")
    print(f"Final weights: {theta}")
    return 0


if '__main__' == __name__:
    main()
