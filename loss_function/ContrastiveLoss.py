import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, enc_features, labels):
        n = labels.shape[0]
        labels = labels.long()
        device = enc_features.device

        similarity_matrix = F.cosine_similarity(enc_features.unsqueeze(1), enc_features.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix, device=device) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask, device=device) - mask
        mask_diagonal = torch.ones(n, n, device=device) - torch.eye(n, n, device=device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_diagonal
        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim  # [batch_size, batch_size]
        no_sim_sum = torch.sum(no_sim, dim=1)  # [batch_size]

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T  # [batch_size, batch_size]
        sim_sum = no_sim_sum_expend
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n, device=device)

        # compute loss of a batch
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)) + 1e-5)
        return loss
