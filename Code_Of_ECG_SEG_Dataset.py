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
            lead_anns, signals = one_signal_preprocessing(numpydataset_path,
                                                          part_ind)
            if lead_anns is None:
                print('lead_anns and signals is None {0}'.format(part_ind))
            else:
                lead_anns_con.append(lead_anns)
                signals_con.append(signals)

        lead_anns_con = np.asarray(lead_anns_con)
        signals_con = np.asarray(signals_con)
        # print('Shape of lead_anns_con : {0} {1}'.format(lead_anns_con.shape[0] * lead_anns_con.shape[1], lead_anns_con.shape[2]))
        # print('Shape of signals_con : {0}'.format(signals_con.shape))
        lead_anns_con = lead_anns_con.reshape(-1, lead_anns_con.shape[2])
        signals_con = signals_con.reshape(-1, signals_con.shape[2])

        # print('Shape of lead_anns_con : {0}'.format(lead_anns_con.shape))
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
        else:
            ecg_dataset = ECG_SEG_Dataset(dataset_path = './lu_dataset/data/')
