import torch
import torch.nn as nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, learnable_bandwidth=False):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        if bandwidth is None:
            self.bandwidth = None
        else:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float)) if learnable_bandwidth else torch.tensor(bandwidth, dtype=torch.float)

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            bandwidth = torch.median(L2_distances)
            return bandwidth
        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        return torch.exp(-L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        n_x = X.size(0)
        n_y = Y.size(0)
        Z = torch.cat([X, Y], dim=0)
        K = self.kernel(Z)

        K_XX = K[:n_x, :n_x]
        K_YY = K[n_x:, n_x:]
        K_XY = K[:n_x, n_x:]

        sum_XX = K_XX.sum() - torch.diag(K_XX).sum()
        sum_YY = K_YY.sum() - torch.diag(K_YY).sum()
        sum_XY = K_XY.sum()

        mmd = sum_XX / (n_x * (n_x - 1) + 1e-8) + sum_YY / (n_y * (n_y - 1) + 1e-8) - 2 * sum_XY / (n_x * n_y + 1e-8)

        return mmd
