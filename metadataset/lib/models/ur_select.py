import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sigmoid = nn.Sigmoid()

def cosine_sim(embeds, prots):
    prots = prots.unsqueeze(0)
    embeds = embeds.unsqueeze(1)
    return F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30)

def apply_selection(features_dict, lambdas, normalize=True):
    """
    Performs masking of features via pointwise multiplying by lambda
    """
    lambdas_01 = sigmoid(lambdas)
    features_list = list(features_dict.values())
    if normalize:
        features_list = [f / (f ** 2).sum(-1, keepdim=True).sqrt()
                         for f in features_list]
    n_cont = features_list[0].shape[0]
    concat_feat = torch.stack(features_list, -1)
    return (concat_feat * lambdas_01)


def sur(context_features_dict, context_labels, max_iter=40):
    """
    SUR method: optimizes selection parameters lambda
    """
    lambdas = torch.zeros([1, 1, len(context_features_dict)]).to("cuda")
    lambdas.requires_grad_(True)
#     n_classes = len(np.unique(context_labels.cpu().numpy()))
#     optimizer = torch.optim.Adadelta([lambdas], lr=(3e+3 / n_classes))
#     for i in range(max_iter):
#         optimizer.zero_grad()
#         selected_features = apply_selection(context_features_dict, lambdas)
#         loss, stat, _ = prototype_loss(selected_features, context_labels,
#                                        selected_features, context_labels)

#         loss.backward()
#         optimizer.step()
    return lambdas


class SurModel(torch.nn.Module):
    
    def __init__(self, classifier, n_domains=8):
        """
        Select Universal Representation
        """
        super(SurModel, self).__init__()
        self._lambda = torch.nn.Parameter(torch.zeros(n_domains)) # feacture select
        self.classifier = classifier

    def forward(self, context_features, context_labels, target_features):
        """
        feature=>feature select=> classifier
        context_features is equivalent to support_features
        target_features is equivalent  to query_features
        """
        # ur
        n_sample, _, _ = context_features.shape
        # ur select
        context_features = F.normalize(context_features, p=2, dim=-1)
        context_features = (torch.sigmoid(self._lambda).unsqueeze(0).unsqueeze(2) * context_features).view(n_sample, -1)
        # ur
        n_sample, _, _ = target_features.shape
        # ur select
        target_features = F.normalize(target_features, p=2, dim=-1)
        target_features = (torch.sigmoid(self._lambda).unsqueeze(0).unsqueeze(2) * target_features).view(n_sample, -1)
        # classisier
        self.classifier.fit(X=context_features, y=context_labels) # compute \mu and \Sigma on support set
        logits = self.classifier.predict(X=target_features)[0]
        torch.cuda.empty_cache()
        
        return logits