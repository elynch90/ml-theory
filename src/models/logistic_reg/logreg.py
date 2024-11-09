from numpy import dot, array, exp, log, mean
from matplotlib import pyplot as plt
import random

EPOCHS = 1024
EPSILON = 1e-9
LEARNING_RATE = 1e-1


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
    if isinstance(y, (float, int)):
        return -y * log(y_hat + epsilon) - (1 - y) * log(1 - y_hat + epsilon)
    return array([-y * log(y_hat + epsilon) - (1 - y) * log(1 - y_hat + epsilon) for y, y_hat in zip(y, y_hat)])


class LogisticRegression:
    def __init__(self, theta=None, bias=0):
        self.theta = theta if theta else [0.1, 0.2]
        self.bias = bias

    def forward(self, X=None) -> float:
        """Forward pass of the model without activation function, aka linear.
        returns the dot product of the weights and the input features
        X: input features
        bias: bias term"""
        return dot(X, self.theta) + self.bias

    def backward(self, x, y, y_hat, m=1) -> float:
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
        # notice how the error is multiplied by y_hat and (1 - y_hat)
        #  this is because the derivative of the sigmoid function is y_hat * (1 - y_hat)
        grads = (1 / m) * dot(x.T, (error_term * y_hat * (1 - y_hat)))
        print(f"Gradients: {grads}")
        return grads

    def get_batches(self, X, Y, batch_size=32):
        batches = []
        for i in range(0, len(X), batch_size):
            batch = (X[i:i+batch_size], Y[i:i+batch_size])
            batches.append(batch)
        return batches

    def stochastic_gradient_descent(
            self, X, Y, batch_size=2**1, epochs=EPOCHS,
            learning_rate=LEARNING_RATE) -> array:
        """Stochastic Gradient Descent (SGD) is an optimization algorithm used
        to minimize some function by iteratively moving in the direction of
        steepest descent as defined by the negative of the gradient.
        X: input features
        Y: target labels
        epochs: number of iterations
        learning_rate: step size"""
        batches = self.get_batches(X, Y, batch_size)
        losses = []
        for epoch in range(epochs):
            # TODO: implement mini-batch gradient descent
            # instead of using a single sample, we can use a batch of samples
            # and take advantage of vectorization to speed up the computation
            batch_losses = []
            for i, (x_batch, y_batch) in enumerate(batches):
                # forward pass
                logits = self.forward(x_batch)
                print(f"\nLogits: {logits}")
                # activation function
                y_hat = sigmoid(logits)
                # y_hat = [int(i > 0.5) for i in y_hat]
                print(f"Predictions: {y_hat}")
                # get the target label (ground truth)
                # calculate loss
                # the loss shows how well the model is doing but
                # is not used to update the weights directly
                loss = loss_func(y_batch, y_hat)
                batch_losses.append(sum(loss) / batch_size)
                # backward pass to calculate the gradients using the error term
                grads = self.backward(x_batch, y_batch, y_hat, m=batch_size)
                # calculate the gradients of the bias term
                bias_grad = (1 / batch_size) * sum(y_hat - y_batch)
                self.bias -= learning_rate * bias_grad / batch_size
                # update weights
                self.theta -= learning_rate * grads
                print(f"Epoch: {epoch}, Loss: {loss}, Error: {loss}, bias: {self.bias}\n")
            epoch_loss = mean(batch_losses)
            losses.append(epoch_loss)
        # plot the loss over the epochs
        plt.plot(losses, label="avg loss per batch over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        return self.theta

    def train(self, X, Y, seed=42) -> array:
        """Train the model using the input features and the target labels
        X: input features
        Y: target labels
        """
        # convert to numpy array
        X = array(X)
        Y = array(Y)
        # ensure n randomized weights for each feature
        features_n = len(X[0])
        self.theta = array([random.random() for _ in range(features_n)])
        # train the model
        self.stochastic_gradient_descent(X, Y)
        return self.theta

    def predict(self, x):
        """Get the output (y-hat) of the model for a given input"""
        logit = self.forward(x)
        prob = sigmoid(logit)
        if isinstance(prob, (float, int)):
            return int(prob > 0.5), prob
        return array([int(p > 0.5) for p in prob]), prob


def main() -> int:
    # Data
    X = [[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    Y = [0, 1, 1, 1, 0, 0, 0]
    X_test = [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]]
    y_test = [0, 1, 1, 1]
    model = LogisticRegression()
    theta = model.train(X, Y)
    # Test
    preds, prob = model.predict(X_test)
    print(f"Predictions: {preds}, Probabilities: {prob}\nground truths: {y_test}")
    accuracy = sum([1 for i, j in zip(preds, y_test) if i == j]) / len(y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Final weights: {theta}")
    return 0


if '__main__' == __name__:
    main()
