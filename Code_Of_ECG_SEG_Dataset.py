import numpy as np
import glob as gb
import sys
import os

class ECG_SEG_Dataset():
    def __init__(self, dataset_path):
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
        header_suffix = '.hea'
        signal_suffix = '.dat'

        part_ind = 1
        part_path = dataset_path + str(part_ind)
        print('part_path : {0}'.format(part_path))

        i_lead = wfdb.rdann(part_path, leads_suffix[0]) 
        ii_lead = wfdb.rdann(part_path, leads_suffix[1])
        iii_lead = wfdb.rdann(part_path, leads_suffix[2])
        avr_lead = wfdb.rdann(part_path, leads_suffix[3])
        avl_lead = wfdb.rdann(part_path, leads_suffix[4])
        avf_lead = wfdb.rdann(part_path, leads_suffix[5])
        v1_lead = wfdb.rdann(part_path, leads_suffix[6])
        v2_lead = wfdb.rdann(part_path, leads_suffix[7])
        v3_lead = wfdb.rdann(part_path, leads_suffix[8])
        v4_lead = wfdb.rdann(part_path, leads_suffix[9])
        v5_lead = wfdb.rdann(part_path, leads_suffix[10])
        v6_lead = wfdb.rdann(part_path, leads_suffix[11])

        print('i_lead : {0}'.format(i_lead.sample))
        
        signal_sample = wfdb.rdsamp(part_path)
        print('signal_sample : {0}'.format(signal_sample))



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
     
        for lead_idx in range(12):
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
            else:
                break
        
        if sample_quality:
            print('Save sample : {0}'.format(part_ind))
            signal_npy_path = numpydataset_path + 'lu_dataset_raw_npy/' + '{0:03d}'.format(part_ind) + '_sig.npy'
            lead_anns_npy_path = numpydataset_path + 'lu_dataset_raw_npy/' + '{0:03d}'.format(part_ind) + '_lead_anns.npy'
            print('signal_npy_path : {0}'.format(signal_npy_path))
            print('lead_anns_npy_path : {0}'.format(lead_anns_npy_path))
            lead_anns = np.asarray(lead_anns)
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

    def show_one_signal(numpydataset_path):
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
        part_ind = 1
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
        # plt.plot(time[500:1001], i_sig[500:1001])
        # plt.plot(time[500:1001], i_ann_sig[500:1001], 'rx')
    
    
    # load_one_signal(dataset_path = './lu_dataset/data/',
    #                 numpydataset_path = './lu_dataset_numpy/',
    #                 part_ind = 9)
    load_signal(dataset_path = './lu_dataset/data/',
                numpydataset_path = './lu_dataset_numpy/')
    # show_one_signal(numpydataset_path = './lu_dataset_numpy/')
    # ecg_dataset = ECG_SEG_Dataset(dataset_path = './lu_dataset/data/')
