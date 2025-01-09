
import torch 
import numpy as np
import torch.nn.functional as F
import scipy.special as sc
import losses

def generate_binomial_mask(B, T, p=0.5):
    tmp  = np.random.binomial(1, p, size=(B, T)) # read https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
    return torch.from_numpy(tmp).to(torch.bool)



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def cropping(x,crop_l=None):
    ts_l = x.size(1) # each instace, how long it is.
    if crop_l is None:
        crop_l = np.random.randint(low=2, high=ts_l+1)
    crop_left = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l
    crop_eleft = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)


    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
    a1 = crop_offset + crop_eleft
    a1_len = crop_right - crop_eleft

    a2 =  crop_offset + crop_left
    a2_len = crop_eright - crop_left

    x1 = take_per_row(x,a1, a1_len)
    x2 = take_per_row(x,a2, a2_len)
    return x1,x2,crop_l



#reasoning: suppose arr:
# [1,2,1,0] and during training it turns into:
# [1,2,0,0], we want to restore the val in index 2 from 0 to its previous value (in this case 1):
# and after filled zero we get -> [1,2,1,0]


#suppose we have an array of loss:

#[3 5 5 10 30 50]
#[5 20 30 10 0 0]
#[3 15 20 30 10 0]
def shift_array_loss(loss_array,n_epoch=3,n_res=6):
    loss_algined = np.zeros((n_epoch,n_res))
    loss_array_flip = np.flip(loss_array,axis=1)
    start_idx = np.sum(loss_array_flip==0,axis=1)
    for i in range(n_epoch):
        loss_algined[i,0:n_res-start_idx[i]]=loss_array_flip[i,start_idx[i]:]
    return loss_algined,start_idx

def shift_loss(loss_per_epoch,n_res=6):
    loss_algined = np.zeros((n_res))
    loss_array_flip = np.flip(loss_per_epoch)
    start_idx = np.sum(loss_array_flip==0)
    loss_algined[0:n_res-start_idx]=loss_array_flip[start_idx:]
    return loss_algined,start_idx



def filled_zero(loss_algined,n_epoch=3):
    for i in range(1,n_epoch):
        loss_algined[i,loss_algined[i,:]==0] = loss_algined[i-1,loss_algined[i,:]==0]
    return loss_algined

#score is an array of scores assigned from the score_func
def pick(score_idx,score):
    #we get the correct index by  subtracting the size of the ( (array - 1) - the target index), and it should get us to the correct index
   correct_indx = (len(score)- 1) - score_idx
   return correct_indx

def pick_score(score):
    x = np.random.multinomial(1, score, size=1)
    return np.where(x == 1)[1][0]



def score_func(loss_algined,start_idx,epoch, a=0.1):
    return sc.softmax(a*(loss_algined[epoch,:-start_idx[epoch]]-loss_algined[epoch-1,:-start_idx[epoch]]))




def loss_selected_resolution(out1,out2,level_sampled,res=10):
    alpha = 0.5
    level = np.ceil(np.log2(out1.shape[1]))+1
    #print('LEVEL: ' + str(level))
    #level_sampled = np.random.randint(0,level)
    #print('LEVEL SAMPLED: ' + str(level_sampled))
    choose_resolution = 2**level_sampled
    print(out1.size())
    print("random choose resultion: "+str(choose_resolution))

    out1 = F.max_pool1d(out1.transpose(1, 2), kernel_size=np.minimum(choose_resolution, out1.shape[1])).transpose(1, 2)

    out2 = F.max_pool1d(out2.transpose(1, 2), kernel_size=np.minimum(choose_resolution, out2.shape[1])).transpose(1, 2)

    print(out1.size())

    print(out2.size())

    loss_select = alpha * losses.instance_contrastive_loss(out1, out2) + (1 - alpha) * losses.temporal_contrastive_loss(out1, out2)
    #Only visualization propose. Not used in training. no_grad means we do not use this part of code to train
    #loss_select += 0.5 * instance_contrastive_loss(out1.mean(), out2)
    return loss_select



def select_res(loss_candid):
    n_epoch,n_res = loss_candid.shape
    loss_aligned, start_idx = shift_array_loss(loss_candid,n_epoch=n_epoch,n_res=n_res)
    loss_aligned = filled_zero(loss_aligned,n_epoch=n_epoch)
    score = score_func(loss_aligned,start_idx,1)
    res_idx_algn = pick_score(score)
    level_sampled = pick(res_idx_algn,score)
    return level_sampled

