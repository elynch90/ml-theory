from numpy import array, dot
import random
from matplotlib import pyplot as plt

LEARNING_RATE = 0.01
EPOCHS = 64


class LinearRegression:
    def __init__(self, solver="ols"):
        self.solver = solver
        # randomly initialize weights
        self.intercept = random.random()
        self.slope = random.random()

    def error(self, Y, y_hats):
        squared_error = (Y - y_hats) ** 2
        return sum(squared_error)

    def gradient_descent(
            self, X, Y, learning_rate=LEARNING_RATE, epochs=EPOCHS):
        tensor_m = len(X)
        for epoch in range(epochs):
            y_hats = self.slope * X + self.intercept
            error = self.error(Y, y_hats)
            # calculate the gradients
            slope_grad = (1 / tensor_m) * sum(X * error)
            intercept_grad = (1 / tensor_m) * error
            print(f"slope_grad: {slope_grad}, intercept_grad: {intercept_grad}")
            # update weights
            self.slope -= learning_rate * slope_grad
            self.intercept -= learning_rate * intercept_grad
            print(
                f"Epoch: {epoch}, Error: {error}, Slope: {self.slope}, Intercept: {self.intercept}")

    def fit(self, X, Y):
        if self.solver == "ols":
            pass
        # use gradient descent
        else:
            self.gradient_descent(X, Y)

    def __repr__(self) -> str:
        return f"Linear Regression Model: y={self.slope}x + {self.intercept}"


def main() -> int:
    # Data
    X = array([1, 2, 3, 4, 5])
    Y = array([2, 3, 4, 5, 6])
    # Model
    model = LinearRegression(solver="grad")
    model.fit(X, Y)
    print(model)
    # Plot
    plt.scatter(X, Y)
    # make a subplot for the regression line
    plt.plot(X, model.slope * X + model.intercept, color="red")
    plt.show()
    return 0


if __name__ == "__main__":
    main()
