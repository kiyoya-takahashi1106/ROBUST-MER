import torch
import torch.nn as nn
import torch.nn.functional as F


def SIMLOSS(lst):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 0
    for i in range(len(lst)-1):
        for j in range(i+1, len(lst)):
            loss += torch.mean(1 - cos(lst[i], lst[j]))
    # return loss
    return loss / ((len(lst)*(len(lst)-1))/2)



def diffloss(input1, input2):
    batch_size = input1.size(0)
    input1 = input1.view(batch_size, -1)
    input2 = input2.view(batch_size, -1)

    # Zero mean
    input1 = input1 - torch.mean(input1, dim=0, keepdims=True)
    input2 = input2 - torch.mean(input2, dim=0, keepdims=True)

    # L2正規化
    input1_l2 = F.normalize(input1, p=2, dim=1)
    input2_l2 = F.normalize(input2, p=2, dim=1)

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    return diff_loss


def DIFFLOSS(s_lst, p_lst):
    loss = 0
    for i in range(len(s_lst)):
        loss += diffloss(s_lst[i], p_lst[i])
    
    for j in range(len(p_lst)-1):
        for k in range(j+1, len(p_lst)):
            loss += diffloss(p_lst[j], p_lst[k])

    # return loss
    return loss / (len(s_lst) + (len(p_lst)*(len(p_lst)-1))/2)



def RECONLOSS(f_lst, r_lst):
    loss = 0
    for i in range(len(f_lst)):
        loss += nn.MSELoss()(f_lst[i], r_lst[i])
    # return loss
    return loss / len(f_lst)
