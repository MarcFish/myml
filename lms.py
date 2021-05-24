import numpy as np
import matplotlib.pyplot as plt
import typing
from sklearn.datasets import make_blobs
from sklearn.preprocessing import minmax_scale
import torch
import torch.nn as nn
from tqdm import tqdm


class LMS(nn.Module):
    def __init__(self, in_features):
        super(LMS, self).__init__()
        self.w = nn.Linear(in_features, 1)

    def forward(self, inputs):
        o = self.w(inputs)
        return o


if __name__ == "__main__":
    dataset = make_blobs(n_samples=500, n_features=2, centers=2)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    model = LMS(2)
    optim = torch.optim.SGD(model.parameters(), 1e-2)

    for epoch in tqdm(range(10)):
        for inputs, labels in zip(*dataset):
            inputs = torch.from_numpy(inputs).float()
            optim.zero_grad()
            pre = model(inputs)
            loss = (labels - pre) ** 2
            loss.backward()
            optim.step()

    hits = 0.
    pre_list = list()
    for inputs, labels in zip(*dataset):
        inputs = torch.from_numpy(inputs).float()
        pre = model(inputs)
        pre_list.append(pre)
        if pre == labels:
            hits += 1
    print(f"accuracy:{hits/dataset[0].shape[0]}")

    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre_list)
    plt.show()
