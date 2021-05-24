import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

def rbf_kernel(x1, x2, sigma=1):
    return torch.exp(-((x1 - x2) ** 2).sum(1)/2*sigma).unsqueeze(0)

def multiquadrics_kernel(x1, x2, sigma=1):
    return torch.sqrt(((x1 - x2) ** 2).sum(1)+sigma**2)

def inv_multiquadrics_kernel(x1, x2, sigma=1):
    return 1 / (multiquadrics_kernel(x1, x2, sigma) + 1e-8)

def laplacian_kernel(x1, x2, sigma=1):
    return torch.exp(-(x1 - x2).sum(1)/2*sigma).unsqueeze(0)

def linear_kernel(x1, x2):
    return torch.matmul(x1, x2.T)

def cos_kernel(x1, x2):
    return torch.tanh(F.cosine_similarity(x1, x2).unsqueeze(0))

def sigmoid_kernel(x1, x2):
    return torch.tanh(linear_kernel(x1, x2))

def polynomial_kernel(x1, x2):
    return (linear_kernel(x1, x2)+1)**3

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.w1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(32, in_features)))
        self.w2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(8, 32)))
        self.w3 = nn.Parameter(nn.init.xavier_normal_(torch.empty(out_features, 8)))

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.unsqueeze(0)
        o1 = rbf_kernel(inputs, self.w1)
        o2 = rbf_kernel(o1, self.w2)
        o3 = linear_kernel(o2, self.w3)
        return o3


EPOCH_SIZE = 10
LR = 1e-3
if __name__ == "__main__":
    dataset = make_circles(n_samples=10000)
    dataset = (minmax_scale(dataset[0]), dataset[1] * 2 - 1)

    model = Model(2, 1)
    opt = torch.optim.Adam(model.parameters(), LR)

    for epoch in range(EPOCH_SIZE):
        with trange(len(dataset[0])) as t:
            for i in t:
                inputs = torch.from_numpy(dataset[0][i]).float()
                labels = torch.tensor(dataset[1][i]).float().reshape(-1)

                model.zero_grad()
                o = model(inputs).flatten()
                loss = ((o - labels)**2)
                loss.backward()
                opt.step()

                t.set_postfix(loss=loss)

    pre_list = []
    for d in dataset[0]:
        o = np.sign(model(torch.from_numpy(d).float()).flatten().detach().numpy())[0]
        pre_list.append(o)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=pre_list)
    plt.show()
