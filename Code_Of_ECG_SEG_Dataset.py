import numpy as np
import glob as gb
import sys
import os
import random
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

class ECG_SEG_Dataset(Dataset):
    def __init__(self, 
                 preprocess_dataset_path,
                 mode='train'):
        signals_con_path = preprocess_dataset_path + 'signals_con.npy'
        lead_cons_path = preprocess_dataset_path + 'lead_anns_con.npy'
        signals = np.load(signals_con_path)
        lead_cons = np.load(lead_cons_path)
        # print('Shape of signals : {0}'.format(signals.shape))
        # print('Shape of lead_cons : {0}'.format(lead_cons.shape))
        num_sample = signals.shape[0]
        if mode == 'train':
            mode_id = np.ones((num_sample), dtype=bool)
            mode_id[9::10] = 0
        else:
            mode_id = np.zeros((num_sample), dtype=bool)
            mode_id[9::10] = 1
        signals = signals[mode_id, :]
        lead_cons = lead_cons[mode_id, :]
        num_sample = signals.shape[0]
        shuffle_id = np.arange(num_sample)
        np.random.shuffle(shuffle_id)
        # print('mode_id : {0}'.format(mode_id))
        # print('Shape of signals : {0}'.format(signals.shape))
        # print('Shape of lead_cons : {0}'.format(lead_cons.shape))
        # print('shuffle_id : {0}'.format(shuffle_id))
        signals = signals[shuffle_id, :]
        lead_cons = lead_cons[shuffle_id, :]
        self.signals = torch.FloatTensor(signals)
        self.lead_cons = torch.LongTensor(lead_cons)

    def __getitem__(self, index):
        random.seed(datetime.time())
        sample_freq = 500 # 500 Hz
        sample_span = 4 # 4 sec
        total_sample_span = 6
        span_len = sample_freq * sample_span
        signals = self.signals[index, :]
        lead_cons = self.lead_cons[index, :]
        stride_start_valid_ind = total_sample_span * sample_freq - span_len
        stride_start_ind = random.randint(a = 0,
                                          b = stride_start_valid_ind)
        # print('Shape of signals : {0}'.format(signals.shape))
        # print('Shape of lead_cons : {0}'.format(lead_cons.shape))
        # print('stride_start_ind : {0}'.format(stride_start_ind))
        signals = signals[stride_start_ind:stride_start_ind + span_len]
        lead_cons = lead_cons[stride_start_ind:stride_start_ind + span_len]
        # print('Shape of signals : {0}'.format(signals.shape))
        # print('Shape of lead_cons : {0}'.format(lead_cons.shape))
        return signals, lead_cons

    def __len__(self):
        return len(self.signals)


if __name__ == '__main__':
    def load_one_signal(dataset_path,
                        numpydataset_path,
                        part_ind = 1):
        import wfdb
        from wfdb import processing
        leads_suffix = {0: 'i',
                        1: 'ii',
                        2: 'iii',
                        3: 'avr',
                        4: 'avl',
                        5: 'avf',
                        6: 'v1',
                        7: 'v2',
                        8: 'v3',
                        9: 'v4',
                        10: 'v5',
                        11: 'v6'}
        syms = {'N': 1, # QRS complex
                'p': 2, # p wave
                't': 3} # t wave
        header_suffix = '.hea'
        signal_suffix = '.dat'
        sample_quality = True

        part_path = dataset_path + str(part_ind)
        print('part_path : {0}'.format(part_path))

        signal_sample = wfdb.rdsamp(part_path)
        signal = signal_sample[0]
        signal_len = signal.shape[0]
        lead_anns = []
        lead_valid = []
     
        for lead_idx in range(12):
            sample_quality = True
            lead_ann_sym = wfdb.rdann(part_path, leads_suffix[lead_idx], return_label_elements=['symbol']).symbol
            lead_ann_time = np.asarray(wfdb.rdann(part_path, leads_suffix[lead_idx]).sample)
            lead_ann_len = len(lead_ann_sym)
            lead_ann = np.zeros(signal_len) 
            lead_ann_groups = lead_ann_len // 3
            # print('lead_ann_sym : {0}'.format(lead_ann_sym))
            # print('lead_ann_time : {0}'.format(lead_ann_time))
            # print('lead_ann_len : {0} lead_ann_groups : {1}'.format(lead_ann_len,
            #                                                         lead_ann_groups))
            for group in range(lead_ann_groups):
                lead_ann_index = group * 3 + 1
                lead_ann_time_start = lead_ann_time[lead_ann_index - 1]
                lead_ann_time_stop = lead_ann_time[lead_ann_index + 1]
                ann_sym = lead_ann_sym[lead_ann_index]
                if ann_sym in syms.keys():
                    ann_sym_class = syms[ann_sym]
                    lead_ann[lead_ann_time_start:lead_ann_time_stop] = ann_sym_class
                    samplle_quality = True
                else:
                    sample_quality = False
                    break
            if sample_quality:
                lead_anns.append(lead_ann)
                lead_valid.append(lead_idx)
            else:
                continue

        # print('lead_valid : {0}'.format(lead_valid))
        sample_quality = (len(lead_valid) != 0)
        # print('sample_quality : {0}'.format(sample_quality))
        
        if sample_quality:
            # print('Save sample : {0}'.format(part_ind))
            signal_npy_path = numpydataset_path + '{0:03d}'.format(part_ind) + '_sig.npy'
            lead_anns_npy_path = numpydataset_path + '{0:03d}'.format(part_ind) + '_lead_anns.npy'
            # print('signal_npy_path : {0}'.format(signal_npy_path))
            # print('lead_anns_npy_path : {0}'.format(lead_anns_npy_path))
            lead_anns = np.asarray(lead_anns)
            # print('lead_anns : {0}'.format(lead_anns.shape))
            # print('signal : {0}'.format(signal.shape))
            signal = signal[:, lead_valid]
            # print('signal : {0}'.format(signal.shape))
            np.save(signal_npy_path, signal)
            np.save(lead_anns_npy_path, lead_anns)
        else:
            print('Sample : {0} isnt good'.format(part_ind))

    def load_signal(dataset_path,
                    numpydataset_path):
        for part_ind in range(1,201):
            load_one_signal(dataset_path,
                            numpydataset_path,
                            part_ind)

    def show_one_signal(numpydataset_path, part_ind):
        import matplotlib.pyplot as plt
        leads_ann_suffix = {0: 'i',
                            1: 'ii',
                            2: 'iii',
                            3: 'avr',
                            4: 'avl',
                            5: 'avf',
                            6: 'v1',
                            7: 'v2',
                            8: 'v3',
                            9: 'v4',
                            10: 'v5',
                            11: 'v6'}
        part_path = numpydataset_path + str(part_ind)

        i_lead_ann_path = part_path + '_' + leads_ann_suffix[0] + '.npy'
        signal_path = part_path + '_sig.npy'

        i_lead_ann = np.load(i_lead_ann_path)
        signal = np.load(signal_path)

        time  = np.arange(signal.shape[0])
        i_sig = signal[:, 0]

        plt.figure()
        plt.plot(time, i_sig)
        plt.plot(time, i_lead_ann, 'rx')

    def one_signal_preprocessing(numpydataset_path,
                                 part_ind):
        part_path = numpydataset_path + '{0:03d}'.format(part_ind)
        lead_anns_path = part_path + '_lead_anns.npy'
        signal_npy_path = part_path + '_sig.npy'
        try:
            lead_anns = np.load(lead_anns_path)
            signal = np.load(signal_npy_path).transpose()
        except:
            print('Exception')
            return None, None
        sample_freq = 500
        valid_sample_start = sample_freq * 2
        valid_sample_end = sample_freq * 8
        # print('Shape of lead_anns : {0}'.format(lead_anns.shape))
        # print('Shape of signal: {0}'.format(signal.shape))
        lead_anns = lead_anns[:, valid_sample_start:valid_sample_end]
        signal = signal[:, valid_sample_start:valid_sample_end]
        # print('Shape of lead_anns : {0}'.format(lead_anns.shape))
        # print('Shape of signal: {0}'.format(signal.shape))
        return lead_anns, signal

    def signal_preprocessing(numpydataset_path,
                             numpydataset_preprocess_path):
        lead_anns_con = []
        signals_con = []

        for part_ind in range(1, 201):
            # print('part_ind : {0}'.format(part_ind))
            lead_anns, signals = one_signal_preprocessing(numpydataset_path,
                                                          part_ind)
            if lead_anns is None:
                print('lead_anns and signals is None {0}'.format(part_ind))
            else:
                if part_ind == 1:
                    lead_anns_con = lead_anns
                    signals_con = signals
                else:
                    lead_anns_con = np.concatenate((lead_anns_con, lead_anns), axis=0)
                    signals_con = np.concatenate((signals_con, signals), axis=0)

        # print('Shape of lead_anns_con : {0} '.format(lead_anns_con.shape))
        # print('Shape of signals_con : {0}'.format(signals_con.shape))
        # print('numpydataset_preprocess_path : {0}'.format(numpydataset_preprocess_path))
        lead_anns_con_path = numpydataset_preprocess_path + 'lead_anns_con.npy'
        signals_con_path = numpydataset_preprocess_path + 'signals_con.npy'
        np.save(lead_anns_con_path, lead_anns_con)
        np.save(signals_con_path, signals_con)

    
    if len(sys.argv) <= 5:
        if len(sys.argv) == 5:
            if sys.argv[1] == 'load_one_signal':
                load_one_signal(dataset_path = sys.argv[2],
                                numpydataset_path = sys.argv[3],
                                part_ind = int(sys.argv[4]))
        elif len(sys.argv) == 4:
            if sys.argv[1] == 'load_signal':
                load_signal(dataset_path = sys.argv[2],
                            numpydataset_path = sys.argv[3])
            elif sys.argv[1] == 'show_one_signal':
                show_one_signal(numpydataset_path = sys.argv[2],
                                part_ind = int(sys.argv[3]))
            elif sys.argv[1] == 'one_signal_preprocessing':
                one_signal_preprocessing(numpydataset_path = sys.argv[2],
                                         part_ind = int(sys.argv[3]))
            elif sys.argv[1] == 'signal_preprocessing':         
                signal_preprocessing(numpydataset_path = sys.argv[2],
                                     numpydataset_preprocess_path = sys.argv[3])
        elif len(sys.argv) == 2:
            ecg_dataset = ECG_SEG_Dataset(preprocess_dataset_path=sys.argv[1])
            sig, lead_ann = ecg_dataset.__getitem__(index = 1)
            sig_npy = sig.numpy()
            lead_ann_npy = lead_ann.numpy()
        #     np.save('./sig_npy.npy', sig_npy)
        #     np.save('./lead_ann.npy', lead_ann_npy)
        # else:
        #     import matplotlib.pyplot as plt
        #     sig_npy = np.load('./sig_npy.npy')
        #     lead_ann_npy = np.load('./lead_ann.npy')
        #     time  = np.arange(sig_npy.shape[0])


        #     plt.figure()
        #     plt.plot(time, sig_npy)
        #     plt.plot(time, lead_ann_npy, 'rx')
