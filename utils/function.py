import torch
import torch.nn as nn


def COSLOSS(lst):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 0
    for i in range(len(lst)-1):
        for j in range(i+1, len(lst)):
            loss += torch.mean(1 - cos(lst[i], lst[j]))
    return loss / ((len(lst)*(len(lst)-1))/2)



def diffloss(input1, input2):
    batch_size = input1.size(0)
    input1 = input1.view(batch_size, -1)
    input2 = input2.view(batch_size, -1)

    # Zero mean
    input1_mean = torch.mean(input1, dim=0, keepdims=True)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)
    input1 = input1 - input1_mean
    input2 = input2 - input2_mean

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    return diff_loss

def DIFFLOSS(s_lst, p_lst):
    loss = 0
    for i in range(len(s_lst)):
        loss += diffloss(s_lst[i], p_lst[i])
    
    for j in range(len(p_lst)-1):
        for k in range(j+1, len(p_lst)):
            loss += diffloss(p_lst[j], p_lst[k])

    return loss / (len(s_lst) + (len(p_lst)*(len(p_lst)-1))/2)



def MSELOSS(f_lst, r_lst):
    loss = 0
    for i in range(len(f_lst)):
        loss += nn.MSELoss()(f_lst[i], r_lst[i])
    return loss / len(f_lst)