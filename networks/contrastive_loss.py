import torch
import torch.nn as nn

class Contrastive_Loss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.07, base_temperature=0.07):
        super(Contrastive_Loss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        #print((features[0]**2).sum())
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        assert labels.shape[0] == batch_size
        mask = torch.eq(labels, labels.T).float().to(self.device)
        #print(labels)
        #print(mask)
        #print('hello', features)
        
        contrast_count = 1
        contrast_feature = features

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        #compute_logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
        #for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast-logits_max.detach()
        
        #print(contrast_feature)

        #tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        #mask out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size*anchor_count).view(-1, 1).to(self.device), 0)
        
        mask = mask*logits_mask

        #compute log prob
        exp_logits = torch.exp(logits)*logits_mask+1e-20

        
        log_prob = logits-torch.log(exp_logits.sum(1, keepdim=True))

        #compute mean of log-likelihood over positive
        #print(mask)
        mean_log_prob_pos = (mask*log_prob).sum(1)/(1+mask.sum(1))

        #print(mean_log_prob_pos)
        loss = -mean_log_prob_pos
        #loss = -(self.temperature/self.base_temperature)*mean_log_prob_pos
        #print(loss.shape)
        #loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        #print(loss)
        #print(loss.shape)
        return loss