import torch
# import geomloss
import torch.nn as nn 
import torch.nn.functional as F

    
# class SinkhornDivergence(nn.Module):

#     def __init__(self, p=2, epsilon=0.1, metric='cosine'):
#         super().__init__()
#         self.loss = geomloss.SamplesLoss(loss="sinkhorn", 
#                                          p=1, 
#                                          blur=epsilon, 
#                                          cost=lambda a, b: self.cost_func(a, b, p=p, metric=metric))

#     def forward(self, X, Y):
#         return self.loss(X, Y)
    
#     def cost_func(self, a, b, p=2, metric='cosine'):
#         """ a, b in shape: (B, N, D) or (N, D)
#         """ 
#         assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
#         if metric=='euclidean' and p==1:
#             return geomloss.utils.distances(a, b)
#         elif metric=='euclidean' and p==2:
#             return geomloss.utils.squared_distances(a, b)
#         else:
#             if a.dim() == 3:
#                 x_norm = a / a.norm(dim=2)[:, :, None]
#                 y_norm = b / b.norm(dim=2)[:, :, None]
#                 M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
#             elif a.dim() == 2:
#                 x_norm = a / a.norm(dim=1)[:, None]
#                 y_norm = b / b.norm(dim=1)[:, None]
#                 M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
#             M = pow(M, p)
#             return M
        

def cosine_cost_conditional(X, Y, labelsX=None, labelsY=None):
    """  
    labelsX: (n,) tensor of int
    labelsY: (m,) tensor of int
    """
    Xn = F.normalize(X, dim=-1)
    Yn = F.normalize(Y, dim=-1)
    C = 1.0 - Xn @ Yn.t()  # cosine similarity

    if labelsX is not None and labelsY is not None:
        # mask different cancer types
        mask = labelsX[:, None] != labelsY[None, :]
        C = C.masked_fill(mask, 1e6)  # large cost

    return C


def sinkhorn_plan(X, Y, eps=0.05, n_iters=100, a=None, b=None, labelsX=None, labelsY=None):
    n, m = X.size(0), Y.size(0)
    if a is None:
        a = torch.full((n,), 1.0 / n, device=X.device)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=Y.device)

    C = cosine_cost_conditional(X, Y, labelsX, labelsY)  # (n, m)

    K = torch.exp(-C / eps)  # exp(-cost/epsilon)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        Kv = K @ v
        u = a / (Kv + 1e-9)
        Ku = K.t() @ u
        v = b / (Ku + 1e-9)

    T = (u[:, None] * K) * v[None, :]
    ot_cost = torch.sum(T * C)
    return T, ot_cost
        

def get_coupling(X, Y, entreg=0.1, p=2, metric='cosine'):
    pass


def soft_info_nce(X, Y, T, labelsX=None, labelsY=None, tau=0.07):
    """
    T: (n, m) OT coupling matrix
    labelsX / labelsY: masks
    """
    with torch.no_grad():
        P = T.clone()
        if labelsX is not None and labelsY is not None:
            mask = labelsX[:, None] != labelsY[None, :]
            P = P.masked_fill(mask, 0.0)  # ??
        P = P / (P.sum(dim=1, keepdim=True) + 1e-9)

    logits = (X @ Y.t()) / tau
    log_prob = logits.log_softmax(dim=1)
    loss_xy = -(P * log_prob).sum(dim=1).mean()

    with torch.no_grad():
        Q = T.clone()
        if labelsX is not None and labelsY is not None:
            mask = labelsX[:, None] != labelsY[None, :]
            Q = Q.masked_fill(mask, 0.0)
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-9)

    logits_yx = (Y @ X.t()) / tau
    log_prob_yx = logits_yx.log_softmax(dim=1)
    loss_yx = -(Q.t() * log_prob_yx).sum(dim=1).mean()

    return 0.5 * (loss_xy + loss_yx)


# def cluster_hard_info_nce(X, Y, labels_X, labels_Y, tau=0.07):
#     """
#     Compute contrastive loss (InfoNCE-like) for X and Y embeddings separately, ensuring
#     embeddings with the same labels are close and embeddings with different labels are distant.

#     Args:
#         X: Tensor of shape (n, d), embeddings for X.
#         Y: Tensor of shape (n, d), embeddings for Y.
#         labels_X: Tensor of shape (n,), labels for X.
#         labels_Y: Tensor of shape (n,), labels for Y.
#         tau: Temperature parameter for scaling logits.

#     Returns:
#         Tuple containing two losses:
#         - loss_X: Contrastive loss for X embeddings.
#         - loss_Y: Contrastive loss for Y embeddings.
#     """
#     # Number of samples
#     n = X.size(0)
    
#     ## ---------------------------------------------------------------------
#     ## Step 1: Similarity logits calculation
#     ## ---------------------------------------------------------------------
#     # Compute pairwise similarity (cosine-like via dot product) for X
#     logits_X = (X @ X.t()) / tau  # (n, n)
    
#     # Compute pairwise similarity (cosine-like via dot product) for Y
#     logits_Y = (Y @ Y.t()) / tau  # (n, n)
    
#     ## ---------------------------------------------------------------------
#     ## Step 2: Generate labels mask for positive samples
#     ## ---------------------------------------------------------------------
#     # Positive sample mask for X: labels_X example pairs with the same label
#     labels_mask_X = (labels_X.unsqueeze(1) == labels_X.unsqueeze(0)).float()  # (n, n)
#     labels_mask_X.fill_diagonal_(0)  # Ignore self-comparisons

#     # Positive sample mask for Y: labels_Y example pairs with the same label
#     labels_mask_Y = (labels_Y.unsqueeze(1) == labels_Y.unsqueeze(0)).float()  # (n, n)
#     labels_mask_Y.fill_diagonal_(0)  # Ignore self-comparisons

#     ## ---------------------------------------------------------------------
#     ## Step 3: Similarity partitioning into positive and negative samples
#     ## ---------------------------------------------------------------------
#     # Exponential logits for X (to scale similarity scores)
#     exp_logits_X = torch.exp(logits_X)

#     # Exponential logits for Y (to scale similarity scores)
#     exp_logits_Y = torch.exp(logits_Y)

#     # Positive similarity scores for X
#     positive_sim_X = (labels_mask_X * exp_logits_X).sum(dim=1)  # (n,)
    
#     # Negative similarity scores for X
#     negative_sim_X = exp_logits_X.sum(dim=1) - positive_sim_X - torch.exp(logits_X.diag())  # subtract diagonal self-similarity

#     # Positive similarity scores for Y
#     positive_sim_Y = (labels_mask_Y * exp_logits_Y).sum(dim=1)  # (n,)
    
#     # Negative similarity scores for Y
#     negative_sim_Y = exp_logits_Y.sum(dim=1) - positive_sim_Y - torch.exp(logits_Y.diag())  # subtract diagonal self-similarity

#     ## ---------------------------------------------------------------------
#     ## Step 4: InfoNCE Loss calculation for X and Y
#     ## ---------------------------------------------------------------------
#     # Loss for X embeddings
#     loss_X = -torch.log(positive_sim_X / (positive_sim_X + negative_sim_X + 1e-8)).mean()

#     # Loss for Y embeddings
#     loss_Y = -torch.log(positive_sim_Y / (positive_sim_Y + negative_sim_Y + 1e-8)).mean()

#     return loss_X, loss_Y


def hard_info_nce(X, Y, tau=0.07):
    """
    Assume X_i matched to Y_i
    """
    n = X.size(0)
    logits = (X @ Y.t()) / tau
    labels = torch.arange(n, device=X.device)
    loss_xy = F.cross_entropy(logits, labels)
    loss_yx = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_xy + loss_yx)


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, learnable_bandwidth=False):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        if bandwidth is None:
            self.bandwidth = None
        else:
            self.bandwidth = nn.Parameter(torch.tensor(bandwidth, dtype=torch.float)) if learnable_bandwidth else torch.tensor(bandwidth, dtype=torch.float) # Make bandwidth learnable
        self.learnable_bandwidth = learnable_bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            # Median heuristic
            n_samples = L2_distances.shape[0]
            # Calculate median instead of mean
            bandwidth = torch.median(L2_distances)
            return bandwidth
        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(L2_distances)
        return torch.exp(-L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
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

        # Unbiased MMD estimator
        sum_XX = K_XX.sum() - torch.diag(K_XX).sum()
        sum_YY = K_YY.sum() - torch.diag(K_YY).sum()
        sum_XY = K_XY.sum()

        mmd = sum_XX / (n_x * (n_x - 1) + 1e-8) + sum_YY / (n_y * (n_y - 1) + 1e-8) - 2 * sum_XY / (n_x * n_y + 1e-8) # Adding small constant for numerical stability

        return mmd


def cosine_similarity_loss(x, y):
    
    cosine_sim = F.cosine_similarity(x, y, dim=-1)
    return 1 - cosine_sim.mean()
