import pickle
import h5py
import json
import numpy as np

from tensorflow.keras.utils import to_categorical

import os


def load_data(data_path):

    mods = ['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']
    snrs = [x for x in range(-20, 20, 2)]

    with open(data_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    X = []
    Y = []
    snr_list = []
    for mod in mods:
        for snr in snrs:
            a = p[(mod, snr)]
            for i in range(p[(mod,snr)].shape[0]):  

                #label.append((mod,snr))
                snr_list.append(snr)
                Y.append(mods.index(mod))

            X.append(a) 

    X = np.vstack(X)
    Y = np.array(Y)
    snr_list = np.array(snr_list)

    return X, Y, snr_list



def create_datasets(
    num_classes, 
    data_path, 
    save_path, 
    idx_path = None, 
):
    # Load data
    X, Y, snr = load_data(data_path)
    Y = to_categorical(Y, num_classes)

    # train:valid:test = 6:2:2
    total_size = X.shape[0]

    if idx_path:
        if idx_path[-1] != '/':
            idx_path += '/'
        valid_idx = np.array(eval(open(idx_path + 'valid_idx.txt').read()))
        test_idx = np.array(eval(open(idx_path + 'test_idx.txt').read()))

        if os.path.exists(idx_path + 'train_idx.txt'):
            print("Trian idx file found. Recovering train idx from the file.")
            train_idx = np.array(eval(open(idx_path + 'train_idx.txt').read()))
        else:
            print("Train idx file not found. Recovering train idx from val idx and test idx.")
            train_idx = np.setdiff1d(np.setdiff1d(np.arange(total_size), valid_idx), test_idx)
    else:
        print("Idx path not specified. Randomly creating train, val, and test idx.")
        train_size = int(total_size / 5 * 3)
        valid_size = int(total_size / 5)
        test_size = total_size - train_size - valid_size

        full_idx = np.random.permutation(total_size)
        [train_idx, valid_idx, test_idx] = np.hsplit(full_idx, [train_size, train_size + valid_size])

    print(train_idx)

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]

    print(X_train.shape, X_valid.shape, X_test.shape)

    Y_train = Y[train_idx]
    Y_valid = Y[valid_idx]
    Y_test = Y[test_idx]

    test_snr = snr[test_idx]

    #-------------------- Save indices -----------------#
    if save_path[-1] != '/':
        save_path += '/'


    f = open(save_path + 'valid_idx.txt', 'w')
    f.write(str(valid_idx.tolist()))
    f.close()

    f = open(save_path + 'test_idx.txt', 'w')
    f.write(str(test_idx.tolist()))
    f.close()

    return (
        (X_train, Y_train), 
        (X_valid, Y_valid), 
        (X_test, Y_test), 
        test_snr
    ) # (Train_set, Valid_set, Test_set, Test_snr)