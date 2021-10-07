import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils
import pdb

def set_loader(opt):
   
    clinical = pd.read_csv('./dataset/Clinical_Variables.csv',index_col=0)
    genetic = pd.read_csv('./dataset/Genetic_alterations.csv',index_col=0)
    surv_time = pd.read_csv('./dataset/Survival_time_event.csv',index_col=0) 
    treatment = pd.read_csv('./dataset/Treatment.csv',index_col=0)

    clinical = clinical.to_numpy()
    genetic = genetic.to_numpy()
    surv_time = surv_time.to_numpy()
    treatment = treatment.to_numpy()
    
    input_data = np.concatenate((clinical,genetic,treatment), axis=1)  
    
    if opt.kfold :
        start = opt.k_index
        print(start)
        test_input = input_data[start*100 : (start+1)*100]
        test_surv = surv_time[start*100 : (start+1)*100]

        indexes = [i for i in range(start*100, (start+1)*100)]
        print(indexes)

        input_data = np.delete(input_data, indexes,0)
        surv_time = np.delete(surv_time , indexes,0)

        input_data = torch.tensor(input_data, dtype=torch.int64)
        surv_time = torch.tensor(surv_time,dtype=torch.float32)

        surv_time = surv_time[:,0].unsqueeze(1)
    
        test_input = torch.tensor(test_input, dtype=torch.int64)
        test_surv = torch.tensor(test_surv, dtype=torch.float32)

        test_surv = test_surv[:,0].unsqueeze(1)
    
        train_dataset = TensorDataset(input_data,surv_time)
        test_dataset = TensorDataset(input_data, surv_time)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
        test_loader = DataLoader(test_dataset, batch_size = opt.batch_size)

        return train_loader, test_loader
    else :
        input_data = torch.tensor(input_data, dtype=torch.int64)
        surv_time = torch.tensor(surv_time,dtype=torch.float32)

        surv_time = surv_time[:,0].unsqueeze(1)
        
        train_dataset = TensorDataset(input_data,surv_time)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)
    
        return train_loader