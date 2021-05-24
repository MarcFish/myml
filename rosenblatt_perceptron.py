import numpy as np
import matplotlib.pyplot as plt
import typing
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale


class RosenblattPerceptron(object):
    def __init__(self, in_features: int, activation: typing.Callable, eta: float) -> None:
        self.in_features = in_features
        self.activation = activation
        self.eta = eta

        self.w = np.random.normal(size=(in_features,)).astype(np.float32)

    def __call__(self, inputs):
        o = self.activation((inputs @ self.w).sum())
        return o

    def train_step(self, inputs, labels):
        loss = labels - self(inputs)
        self.w = self.w + self.eta * loss * inputs
        return loss


if __name__ == "__main__":
    dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    model = RosenblattPerceptron(2, np.sign, 1e-2)

    for inputs, labels in zip(*dataset):
        loss = model.train_step(inputs, labels)

    hits = 0.
    pre_list = list()
    for inputs, labels in zip(*dataset):
        pre = model(inputs)
        pre_list.append(pre)
        if pre == labels:
            hits += 1
    print(f"accuracy:{hits/dataset[0].shape[0]}")

    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre_list)
    plt.plot(np.linspace(0, 1, num=100), -model.w[0]/model.w[1]*np.linspace(0, 1, num=100))
    plt.show()
