import numpy as np
import os
import datetime
import random
import torch
import torch.nn as nn
import torch.quantization as quantization

import Code_Of_ECG_SEG_Dataset as DS
import Code_Of_ECG_SEG_model as model
from torch.utils.data import Dataset, DataLoader

def get_device(quantized_switch=False):
    if quantized_switch:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def train(train_set, valid_set, model, train_config, device):
    n_epochs = train_config['n_epochs']
    optimizer = getattr(torch.optim, train_config['optimizer'])(model.parameters(),
                                                               **train_config['optim_hparas'])
    min_loss = 1000.0
    running_loss = 0.0
    loss_record = {'train': [],
                   'valid': [],
                   'running_loss': [],
                   'running_epoch': [],
                   'dev_epoch': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        print('epoch : {0}'.format(epoch))
        running_loss = 0
        model.train()
        for signals, lead_label in train_set:
            optimizer.zero_grad()
            signals, lead_label = signals.to(device), lead_label.to(device)
            pre_label = model(signals)
            print('Size of signals : {0}'.format(signals.size()))
            print('Size of lead_label : {0}'.format(lead_label.size()))
            print('Size of pre_label : {0}'.format(pre_label.size()))
        epoch += 1

def main():
    load_model = False
    retrain = True
    quantized_switch = False

    model_name = './models/ecg_seg_model.pt'
    quantized_model_name = './models/ecg_seg_model_q.pt'
    lu_dataset_suffix = './lu_dataset_numpy/lu_dataset_preprocess_npy/'

    batch_size = 32

    encoder_parameters = {'input_channel': [1, 4, 8, 16, 32],
                          'middle_channel': [4, 8, 16, 32, 64],
                          'output_channel': [4, 8, 16, 32, 64],
                          'kernel_size': 9,
                          'padding': 4,
                          'stride': 1,
                          'padding_mode': 'zeros'}
    pool_parameters = {'kernel_size': 8,
                       'stride': 2,
                       'padding': 3}
    tran_conv_parameters = {'input_channel': [64, 32, 16, 8],
                            'output_channel': [64, 32, 16, 8],
                            'kernel_size': 8,
                            'stride': 2,
                            'padding': 3}
    decoder_parameters = {'input_channel': [96, 48, 24, 12],
                          'middle_channel': [32, 16, 8, 4],
                          'output_channel': [32, 16, 8, 4],
                          'kernel_size': 3,
                          'stride': 1,
                          'padding': 1,
                          'padding_mode': 'zeros'}
    train_config = { 'n_epochs': 1,
                     'batch_size': 1,
                     'optimizer': 'Adagrad',
                     'optim_hparas': { 'lr': 0.005,
                                       'lr_decay': 0.001},
                     'early_stop': 200,
                     'save_path': model_name}
    train_dataset = DS.ECG_SEG_Dataset(preprocess_dataset_path = lu_dataset_suffix,
                                    mode = 'train')
    valid_dataset = DS.ECG_SEG_Dataset(preprocess_dataset_path = lu_dataset_suffix,
                                    mode = 'valid')
    test_dataset = DS.ECG_SEG_Dataset(preprocess_dataset_path = lu_dataset_suffix,
                                      mode = 'test')

    # random_seed = datatime.time()
    random_seed = 100
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = get_device(quantized_switch)
    print('Device : {0}'.format(device))

    train_set = DataLoader(train_dataset, batch_size, shuffle = True, drop_last = False,
                           num_workers = 0, pin_memory = True)
    valid_set = DataLoader(valid_dataset, batch_size, shuffle = True, drop_last = False,
                           num_workers = 0, pin_memory = True)
    test_set = DataLoader(valid_dataset, batch_size, shuffle = True, drop_last= False,
                          num_workers = 0, pin_memory = True)
   
    if load_model:
        print('Not support temporarily')
    else:
        ecg_unet_model = model.unet_1d_model(encoder_parameters = encoder_parameters,
                                             pool_parameters = pool_parameters,
                                             decoder_parameters = decoder_parameters,
                                             tran_conv_parameters = tran_conv_parameters).to(device)
    
    if ecg_unet_model is not None:
        if retrain:
            train(train_set = train_set,
                  valid_set = valid_set,
                  model = ecg_unet_model,
                  train_config = train_config,
                  device = device)
        else:
            print('Nout supports')
    else:
        print('Model does\' exist')


if __name__ == '__main__':
    main()
