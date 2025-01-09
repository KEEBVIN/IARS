import torch
import torch.nn.functional as F

import numpy as np

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss



def loss(out1,out2,w=None,res=10):
    alpha = 0.5
    if w is None:
        w = np.ones((res))/res
    loss = torch.tensor(0., device=out1.device)
    #print("intialize loss value to: "+str(loss))
    d = 0
    S=np.zeros((res,1))
    while out1.size(1) > 1:

        #print("compute loss value until resolution: "+str(d))
        loss_cur_level = alpha * instance_contrastive_loss(out1, out2) + (1 - alpha) * temporal_contrastive_loss(out1, out2)

        loss += w[d]*loss_cur_level  #the instance contrastive loss
        S[d] =loss_cur_level.item()
        #loss += (1 - alpha) * temporal_contrastive_loss(out1, out2)  #the temporal contrastive loss 
        #print("loss value: "+str(loss))
        d = d + 1
        out1 = F.max_pool1d(out1.transpose(1, 2), kernel_size=2).transpose(1, 2) # Read this: https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html
        out2 = F.max_pool1d(out2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if out1.size(1) == 1:
        loss += w[d] * alpha * instance_contrastive_loss(out1, out2)
        d = d + 1
    return loss,S


# def loss(out1,out2,w=None,res=12):
#     alpha = 0.5
#     if w is None:
#         w = np.ones((res))/res
#     loss = torch.tensor(0., device=out1.device)
#     #print("intialize loss value to: "+str(loss))
#     d = 0
#     S=np.zeros((res,1))
#     while out1.size(1) > 1:
#         #print("compute loss value until resolution: "+str(d))
#         loss_cur_level = alpha * instance_contrastive_loss(out1, out2) + (1 - alpha) * temporal_contrastive_loss(out1, out2)

#         loss += w[d]*loss_cur_level  #the instance contrastive loss 
#         S[d] =loss_cur_level.item()
#         #loss += (1 - alpha) * temporal_contrastive_loss(out1, out2)  #the temporal contrastive loss 
#         #print("loss value: "+str(loss))
#         d = d + 1
#         out1 = F.max_pool1d(out1.transpose(1, 2), kernel_size=2).transpose(1, 2) # Read this: https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html
#         out2 = F.max_pool1d(out2.transpose(1, 2), kernel_size=2).transpose(1, 2)

#     if out1.size(1) == 1:
#         loss += w[d] * alpha * instance_contrastive_loss(out1, out2)
#         d = d + 1
#     return loss,S


def loss_random(out1,out2,w=None,res=10):
    alpha = 0.5
    level = np.ceil(np.log2(out1.shape[1]))+1
    print('LEVEL: ' + str(level))
    level_sampled = np.random.randint(0,level)
    print('LEVEL SAMPLED: ' + str(level_sampled))
    choose_resolution = 2**level_sampled
    print(out1.size())
    print("random choose resultion: "+str(choose_resolution))

    out1 = F.max_pool1d(out1.transpose(1, 2), kernel_size=np.minimum(choose_resolution, out1.shape[1])).transpose(1, 2)

    out2 = F.max_pool1d(out2.transpose(1, 2), kernel_size=np.minimum(choose_resolution, out2.shape[1])).transpose(1, 2)

    print(out1.size())

    print(out2.size())

    loss_random = alpha * instance_contrastive_loss(out1, out2) + (1 - alpha) * temporal_contrastive_loss(out1, out2)
    #Only visualization propose. Not used in training. no_grad means we do not use this part of code to train


    return loss_random



