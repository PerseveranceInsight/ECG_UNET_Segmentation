import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    signal = np.load('./signal_npy.npy')
    lead_label = np.load('./lead_label_npy.npy')
    pred_label = np.load('./pred_label_npy.npy')

    signal = signal.reshape((1, 2000))
    lead_label = lead_label.reshape((1, 2000))
    pred_label = pred_label.reshape((1, 2000))

    time = np.arange(0, 2000).reshape((1, 2000))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, 'g')
    # plt.plot(time, lead_label, 'rx')
    plt.subplot(2, 1, 2)
    plt.plot(time, signal, 'gs')
    # plt.plot(time, pred_label, 'bo')
    plt.show()

