import numpy as np


class SOM(object):
    def __init__(self, x, y, in_features, sigma=1.0, lr=0.2):
        self.x = x
        self.y = y
        self.in_features = in_features

        self.weights = np.random.normal(size=(x, y, in_features))
        self.weights = self.weights / np.linalg.norm(self.weights, axis=-1, keepdims=True)

        self.sigma = sigma
        self.lr = lr

        xx, yy = np.meshgrid(np.arange(x), np.arange(y))
        self.xy = np.stack([yy, xx], axis=-1)

    def train(self, inputs, num_iter):
        for i, d in enumerate(np.arange(num_iter) % len(inputs)):
            x = inputs[d]
            sigma = self._decay(self.sigma, i)
            eta = self._decay(self.lr, i)
            h = self._neighborhood(self.winner(x), sigma)
            self.weights += np.einsum("ij, ijk->ijk", eta * h, x - self.weights)

    def _neighborhood(self, win, sig):
        d = 2 * sig * sig
        o = np.exp(-np.power(self.xy - self.xy[win], 2).sum(-1)/d)
        return o

    def _distance(self, x, y):
        return np.linalg.norm(x - y, axis=-1)

    def _activate(self, x):
        return self._distance(x, self.weights)

    def winner(self, x):
        act_map = self._activate(x)
        return np.unravel_index(act_map.argmin(), act_map.shape)

    def _decay(self, init, num_iter, rate=1e-2):
        return init * np.exp(-num_iter * rate)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import typing
    from sklearn.datasets import make_blobs, make_circles
    from sklearn.preprocessing import minmax_scale
    from collections import defaultdict, Counter
    from sklearn.model_selection import train_test_split
    from itertools import product

    # classification
    # dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    dataset = make_circles(n_samples=500)
    dataset = (minmax_scale(dataset[0]), dataset[1])
    train_data, test_data, train_label, test_label = train_test_split(*dataset)

    model = SOM(10, 10, 2)
    model.train(train_data, 500)

    win_map = defaultdict(list)
    for d, l in zip(train_data, train_label):
        win_map[model.winner(d)].append(l)
    for pos in win_map:
        win_map[pos] = Counter(win_map[pos])

    hits = 0.
    pre_list = list()
    for inputs, labels in zip(test_data, test_label):
        win_pos = model.winner(inputs)
        if win_pos not in win_map:
            pre = np.sum(list(win_map.values())).most_common()[0][0]
        else:
            pre = win_map[win_pos].most_common()[0][0]
        pre_list.append(pre)
        if pre == labels:
            hits += 1
    print(f"accuracy:{hits/test_data.shape[0]}")


    # clustering
    dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    # dataset = make_circles(n_samples=500)
    dataset = (minmax_scale(dataset[0]), dataset[1])

    model = SOM(2, 2, 2)
    model.train(dataset[0], 5000)
    pre_list = []
    for inputs in dataset[0]:
        pre = np.ravel_multi_index(model.winner(inputs), (2, 2))
        pre_list.append(pre)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre_list)
    plt.show()

    img = np.random.uniform(low=0., high=1., size=(500, 500, 3))
    pixels = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    model = SOM(4, 4, 3)
    model.train(pixels, 5000)

    pre = np.zeros(img.shape)
    for i, j in product(range(img.shape[0]), range(img.shape[0])):
        winner = model.winner(img[i, j])
        pre[i, j] = model.weights[winner]

    # img = (img * 255).astype(np.uint8)
    # pre = (pre * 255).astype(np.uint8)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(pre)
    # plt.show()