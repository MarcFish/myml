import numpy as np


class PCA(object):
    def __init__(self, in_features: int, out_features: int, eta: float) -> None:
        self.weights = np.random.normal(loc=0.0, scale=0.1, size=(in_features, out_features))
        self.eta = eta

    def __call__(self, inputs):
        return inputs @ self.weights

    def train_step(self, inputs):
        y = self(inputs)
        for j in range(self.weights.shape[1]):
            x_ = inputs - (self.weights[:, :j] * y[:j]).sum(-1)
            delta_w = self.eta * y[j] * x_ - self.eta * y[j] * y[j] * self.weights[:, j]
            self.weights[:, j] += delta_w


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import typing
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import minmax_scale

    dataset = make_blobs(n_samples=500, n_features=10, centers=2)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    model = PCA(10, 2, 0.02)

    for inputs in dataset[0]:
        model.train_step(inputs)

    pre = model(dataset[0])
    plt.scatter(pre[:, 0], pre[:, 1], c=dataset[1])
    plt.show()
