import numpy as np
from sklearn.datasets import make_blobs
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    model = linear_model.LinearRegression()
    model.fit(dataset[0], dataset[1])
    model.score(dataset[0], dataset[1])

    model2 = linear_model.Ridge()
    model2.fit(dataset[0], dataset[1])
    model2.score(dataset[0], dataset[1])

    hits = 0.
    pre_list = list()
    for inputs, labels in zip(*dataset):
        pre = model.predict(inputs.reshape(1,-1))
        pre_list.append(pre)
        if pre == labels:
            hits += 1
    print(f"accuracy:{hits/dataset[0].shape[0]}")

    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre_list)
    # plt.plot(np.linspace(0, 1, num=100), -model.coef_[1]/model.coef_[0]*np.linspace(0, 1, num=100))
    plt.show()
