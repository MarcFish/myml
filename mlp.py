import numpy as np
import matplotlib.pyplot as plt
import typing
from sklearn.datasets import make_circles
from sklearn.preprocessing import minmax_scale
import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import trange


BATCH_SIZE = 32
EPOCH_SIZE = 10
LR = 1e-2

if __name__ == "__main__":
    dataset = make_circles(n_samples=50000)
    dataset = (minmax_scale(dataset[0]), dataset[1])

    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.Sigmoid(),
        nn.Linear(32, 32),
        nn.Sigmoid(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    opt = torch.optim.Adam(model.parameters(), LR)

    for epoch in range(EPOCH_SIZE):
        with trange(len(dataset[0])//BATCH_SIZE) as t:
            for i in t:
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                batch = torch.from_numpy(dataset[0][start:end]).float()
                batch_label = torch.from_numpy(dataset[1][start:end]).float()

                model.zero_grad()
                o = model(batch).flatten()
                loss = F.binary_cross_entropy(o, batch_label)
                loss.backward()
                opt.step()

                t.set_postfix(loss=loss)


    o = model(torch.from_numpy(dataset[0]).float()).flatten().detach().numpy()
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=o)
    # plt.plot(np.linspace(0, 1, num=100), -model.coef_[1]/model.coef_[0]*np.linspace(0, 1, num=100))
    plt.show()
