# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
#separate files with each process
import losses
import plotting 
import classifiers
import augments
import encoders
import run
import os

import argparse



def training(args):

#Available testing dataset: BasicMotions,ArticularyWordRecognition,UWaveGestureLibrary,CharacterTrajectories,NATOPS,HandMovementDirection,SpokenArabicDigits,
#Additional datasets: StandWalkJump, SelfRegulationSCP1, SelfRegulationSCP2, RacketSports, MotorImagery, LSST, Libras, JapaneseVowels, Heartbeat
#continued: Handwriting, FingerMovements, EthanolConcentration, Epilepsy, DuckDuckGeese, Cricket, AtrialFibrilation, Phoneme,EigenWorms,PEMS-SF

#datasets used for papers(smallest to largest length): HandMovementDirection, Heartbeat, AtrialFibrilation, SelfRegulationSCP1,
# Phoneme, SelfRegulationSCP2, Cricket, EthanolConcentration, StandWalkJump, MotorImagery
    data_set = args.ds
    
    #directory to save figures
    fig_dir = args.dir+"/" + data_set


    train_data = np.load('IARS/Datasets/'+data_set+'/X_train.npy')
    train_label = np.load('IARS/Datasets/'+data_set+'/y_train.npy')





    x = torch.from_numpy(train_data).float().cuda()



    #array to store the proposed accuracy and time
    proposed_acc_arr = []
    proposed_time_arr = []

    """#Data Loader"""

    class SimpleTSDataset2(Dataset):
        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx,:,:]

    # np.save('a.npy',X.detach().cpu().numpy())

    dataset2 = SimpleTSDataset2(x)

    print(x.shape)

    train_dataloader = DataLoader(dataset2, batch_size=32, shuffle=True)

    for x_tmp in train_dataloader:
        print(x_tmp.shape)

    #optional reshaping
    # x_stream = x_tmp.view(x.shape[0]*x.shape[1],-1)

    # x_stream.unsqueeze(dim=0)

    # x_stream.shape

    # X = x_stream.unfold(0, x.shape[1], 1)

    # X = X.transpose(1,2)

    # X.shape

    #Proposed implementation

    #testing 5 times
    for experiment in range(args.i):
        start = time.time()


        net = encoders.TSEncoder(input_dims=x.shape[2], output_dims=32, hidden_dims=64, depth=2)
        net = net.cuda()

        res = np.log2(x.shape[1])
        res = res.astype('int')
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
        total = 0
        avg = 0
        array_loss_proposed = []
        loss_prev = None
        loss_curr = []


        for epoch in range(args.e):
            for batch_part in train_dataloader:
                if loss_prev is not None:
                    loss_candid = np.concatenate((loss_prev,loss_curr),axis=1).T
                    level_sampled = augments.select_res(loss_candid)
                    #loss_select = loss_selected_resolution(out1,out2,level_sampled,res=10)
                else:
                    level_sampled = np.random.randint(0,res)
                x1,x2,l = augments.cropping(batch_part)
                print(l)
                net.train()
                optimizer.zero_grad()
                out1 = net(x1)
                out2 = net(x2)
                out1 = out1[:,-l:,:]
                out2 = out2[:, :l,:]
                with torch.no_grad():
                    loss_value, loss_curr = losses.loss(out1,out2,res=int(np.ceil(np.log2(batch_part.shape[1]))))
                loss_train = augments.loss_selected_resolution(out1,out2,level_sampled,res=10)
                loss_prev = np.copy(loss_curr)
                array_loss_proposed.append(loss_curr)
                loss_train.backward()
                optimizer.step()
                print(loss_curr)
        end = time.time()
        print(end - start)
        running_time = end - start

        #PLOT EMBEDDING
        #####
        #####
        ####
        ###


            
        test_data = np.load('IARS/Datasets/'+data_set+'/X_test.npy')
        test_label = np.load('IARS/Datasets/'+data_set+'/y_test.npy')

        model_used = 'IARS'

        #plot embedding
        test_data_tensor = torch.from_numpy(test_data)
        all_data = torch.cat((x,test_data_tensor.cuda())).float()
        all_label = np.concatenate((train_label,test_label),axis=0)
        #Without average pooling
        plotting.tsne_plot(net,all_data,all_label,model_used,fig_dir,experiment,data_set)


        #SECOND

        all_repr = plotting.tsne_plot_avg(net,all_data,all_label,model_used,fig_dir,experiment,data_set)



        split = x.shape[0]
        train_repr = all_repr[:split,:]
        test_repr = all_repr[split:,:]



        #accuracy
        test_repr = test_repr.cpu().detach().numpy()
        train_repr = train_repr.cpu().detach().numpy()
        acc_proposed = classifiers.eval_classification(train_repr, train_label, test_repr, test_label)

        #keep track of each accuracy per iteration
        proposed_acc_arr.append(acc_proposed)
        proposed_time_arr.append(running_time)



    #Run Originial ts2vec

    original_acc_arr = []
    original_time_arr = []


    fig_dir_ts2vec = args.dir+'/' + data_set
    model_used = 'ts2vec'

    for original_experiment in range(args.i):
        start = time.time()
        net2 = encoders.TSEncoder(input_dims=x.shape[2], output_dims=32, hidden_dims=64, depth=2)
        net2 = net2.cuda()

        res = np.log2(x.shape[1])
        res = res.astype('int')
        optimizer = torch.optim.AdamW(net2.parameters(), lr=args.lr)
        total = 0
        avg = 0
        array_loss_all = []
        prev_resolution_loss = None
        curr_resolution_loss = []
        for i in range(args.e):
            for batch_part in train_dataloader:
                net.train()
                optimizer.zero_grad()
                x1,x2,l = augments.cropping(batch_part)
                #x1 = x1.cuda()
                #x2 = x2.cuda()
                out1 = net2(x1)
                out2 = net2(x2)
                out1 = out1[:,-l:,:]
                out2 = out2[:, :l,:]
                #changed res to res-1 due to out of bounds error, size 9 but trying to reach index 9 rather than 8.
                #l_h = loss_random(out1,out2,None,10)
                #with torch.no_grad():
                lss, curr_resolution_loss = losses.loss(out1,out2,res=int(np.ceil(np.log2(batch_part.shape[1]))))


                    #flip loss and align
                    #fill zero
                    #compute score
                    #pick resolution
                    #switch back to original idxlo
                    #compute loss_select


                lss.backward()
                #l_h.backward()
                #loss_select.backward()
                optimizer.step()
                #prev_resolution_loss = np.copy(curr_resolution_loss)
                #with torch.no_grad():
                print(curr_resolution_loss)
                array_loss_all.append(curr_resolution_loss)



            #with torch.no_grad():


            #print('loss value: '+str(l_h.item()))

        end = time.time()
        print(end - start)
        running_time_original = end-start

        # #plot embedding
        test_data_tensor = torch.from_numpy(test_data)
        all_data = torch.cat((x,test_data_tensor.cuda())).float()
        all_label = np.concatenate((train_label,test_label),axis=0)
        #Without average pooling

        # all_data = torch.from_numpy(all_data).cuda()
        # all_label = torch.from_numpy(all_label).cuda()

        plotting.tsne_plot(net2,all_data,all_label,model_used,fig_dir_ts2vec,original_experiment,data_set)
        #adding code to save this figure
        #tsne_[dataset_name]_without_pool_original.png

        all_repr = plotting.tsne_plot_avg(net2,all_data,all_label,model_used,fig_dir_ts2vec,original_experiment,data_set)
        #adding code to save this figure
        #tsne_[dataset_name]_avgpool_original.png

        split = x.shape[0]
        train_repr = all_repr[:split,:]
        test_repr = all_repr[split:,:]

        test_repr = test_repr.cpu().detach().numpy()
        train_repr = train_repr.cpu().detach().numpy()
        acc_orig = classifiers.eval_classification(train_repr, train_label, test_repr, test_label)

        original_acc_arr.append(acc_orig)
        original_time_arr.append(running_time_original)

    all_data.shape

    all_label.shape

    #proposed accuracy and time
    print('PROPOSED ACC:')
    print(proposed_acc_arr)
    print('PROPOSED TIME:')
    print(proposed_time_arr)

    #original accuracy and time
    print('ORIGINAL ACC')
    print(original_acc_arr)
    print('ORIGINAL TIME:')
    print(original_time_arr)

    return None