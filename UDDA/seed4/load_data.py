import os, sys
import torch
import numpy as np
import scipy.io as scio

def get_data_path(file_path):
    data_path = []
    for f in os.listdir(file_path):
        if f.startswith("."):
            continue
        else:
            data_path.append(os.path.join(file_path, f))
    
    return data_path


def load_trained_data(samples_path_list, labels_list):
    # load the label data
    # print("The groundtrue:", labels_list)

    train_sample = None
    train_label = None
    for path in samples_path_list:
        # load the sample data
        sample = scio.loadmat(path, verify_compressed_data_integrity=False)
        # print("subject's path: ",path)
        flag = 0
        for key, val in sample.items():
            if key.startswith("de_LDS"):
                # print("key: {}, value'shape: {}".format(key, val.shape))
                if train_sample is None:
                    train_sample = val
                    train_label = np.full(val.shape[1], labels_list[flag])
                else:
                    train_sample = np.concatenate((train_sample, val), axis=1)
                    train_label = np.concatenate((train_label, np.full(val.shape[1], labels_list[flag])), axis=0)
                flag += 1

    return train_sample, train_label

def normalize(features, select_dim=0):
    features_min, _ = torch.min(features, dim=select_dim)
    features_max, _ = torch.max(features, dim=select_dim)
    features_min = features_min.unsqueeze(select_dim)
    features_max = features_max.unsqueeze(select_dim)
    return (features - features_min)/(features_max - features_min)

def reshape_feature(fts_sample):
    fts = None
    if torch.is_tensor(fts_sample):
        elec, number, freq = fts_sample.shape
        fts = fts_sample.permute(1, 0, 2).reshape(number, elec * freq)
    else:
        print("Something Wrong!!!")

    return fts


def load4train(samples_path_list, labels_list):
    """
    load the SEED data set for TL
    """
    train_sample, train_label = load_trained_data(samples_path_list, labels_list)
    # transfer from ndarray to tensor
    train_sample = torch.from_numpy(train_sample).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)
    # normalize tensor
    train_sample = normalize(train_sample)
    # reshape the tensor
    train_sample = reshape_feature(train_sample)
    
    return train_sample, train_label


if __name__ == "__main__":
    session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    label_lists = [session1_label, session2_label, session3_label]

    config_path = {"file_path": "/home/EGG_Data/SEED4/eeg_feature_smooth/3/"}
    path_list = get_data_path(config_path["file_path"])
    path_list.remove(config_path["file_path"]+"1_20161126.mat")
    target_path_list = [config_path["file_path"]+"1_20161126.mat"]
    source_path_list = path_list # [path_list.pop()]
    # print("target path: ", target_path_list)
    # print("source path: ", source_path_list, len(source_path_list))
    # Data loader
    source_sample, source_label = load4train(source_path_list, label_lists[2])
    print("source sample shape: {}, label shape: {}".format(source_sample.shape, source_label.shape))
    target_sample, target_label = load4train(target_path_list, label_lists[2])
    print("target sample shape: {}, label shape: {}".format(target_sample.shape, target_label.shape))