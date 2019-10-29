import torch
import scipy

def weighted_mse_loss(input,target):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    weights = Variable(torch.Tensor([0.5,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15,0.5/15])).cuda()
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean()
    return loss

def regression_loss(input, targets):
    pass

def classification_loss(input, targets):
    torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    pass

def joint_loss(optimized_points):
    pass