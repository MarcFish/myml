import numpy as np
from sklearn.datasets import make_circles
from sklearn import linear_model
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel

if __name__ == "__main__":
    dataset = make_circles(n_samples=500)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    cluster = KMeans(n_clusters=10)
    cluster.fit(dataset[0])

    o = pairwise_kernels(dataset[0], cluster.cluster_centers_, "rbf")

    model = linear_model.LinearRegression()
    model.fit(o, dataset[1])
    pre = model.predict(o)

    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre)
    # plt.plot(np.linspace(0, 1, num=100), -model.coef_[1]/model.coef_[0]*np.linspace(0, 1, num=100))
    plt.show()
