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

def valid(valid_set, model, device):
    model.eval()
    valid_loss = 0
    for signal, lead_label in valid_set:
        signal, lead_label = signal.to(device), lead_label.to(device)
        with torch.no_grad():
            pred_label = model(signal)
            loss = model.cal_loss(pred_label, lead_label)
        valid_loss += loss.detach().cpu().item() 
    valid_loss /= len(valid_set)
    return valid_loss

def pred(data_set, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    signal, lead_label = data_set.__getitem__(index = 10)
    signal_npy = signal.numpy()
    signal = signal.view((1, 1, 2000))
    signal = signal.to(device)
    pred_label = model.forward(signal)
    pred_label = softmax(pred_label)
    pred_label = torch.argmax(pred_label, dim=1)
    
    pred_label_npy = pred_label.cpu().detach().numpy()
    lead_label_npy = lead_label.numpy()
    np.save('./signal_npy.npy', signal_npy)
    np.save('./pred_label_npy.npy', pred_label_npy)
    np.save('./lead_label_npy.npy', lead_label_npy)
    print(np.where(pred_label_npy != lead_label_npy))
    print(np.where(pred_label_npy != lead_label_npy)[1].shape)


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
            loss = model.cal_loss(pre_label, lead_label)
            loss.backward()
            optimizer.step()
            running_loss += loss
            loss_record['train'].append(loss.detach().cpu().item())
        running_loss /= len(train_set)
        loss_record['running_loss'].append(running_loss.detach().cpu().item())
        loss_record['running_epoch'].append(epoch)
        valid_loss = valid(valid_set, model, device)
        # print('valid_loss : {0}'.format(valid_loss))
        if valid_loss < min_loss:
            min_loss = valid_loss
            print('Saving model (epoch {:4d}, valid_loss : {:.4f})'.format(epoch+1, min_loss))
            torch.save(model.state_dict(), train_config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        if early_stop_cnt > train_config['early_stop']:
            break

def main():
    load_model = True
    retrain = False
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
    train_config = { 'n_epochs': 1000,
                     'batch_size': batch_size,
                     'optimizer': 'Adagrad',
                     'optim_hparas': { 'lr': 0.015,
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
    valid_set = DataLoader(valid_dataset, 1, shuffle = True, drop_last = False,
                           num_workers = 0, pin_memory = True)
    test_set = DataLoader(test_dataset, 1, shuffle = True, drop_last= False,
                          num_workers = 0, pin_memory = True)
   
    if load_model:
        ecg_unet_model = model.unet_1d_model(encoder_parameters = encoder_parameters,
                                             pool_parameters = pool_parameters,
                                             decoder_parameters = decoder_parameters,
                                             tran_conv_parameters = tran_conv_parameters).to(device)
        ecg_unet_model.load_state_dict(torch.load(train_config['save_path']))
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
            pred(data_set = test_dataset,
                 model = ecg_unet_model,
                 device = device)
    else:
        print('Model does\' exist')


if __name__ == '__main__':
    main()
