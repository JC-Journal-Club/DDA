import os
import re
import numpy as np
import scipy.io as scio

#def get_data_path():
#    '''
#    @Description: get the path of data directory
#    '''
#    parent_path = os.path.abspath("..")
#    data_path = os.path.abspath(os.path.join(parent_path, "SEED", "ExtractedFeatures"))
#    
#    return data_path

all_samples_path = []
def get_all_files(current_path):
    '''
    @Description: get all mat files
    '''
    labels_path = None
    list_files = os.listdir(current_path)
    for f in list_files:
        # ignore the hidden file
        if f.startswith("."):
            continue
        file_path = os.path.join(current_path, f)
        # recursive traversal of all files.
        if os.path.isdir(file_path):
            get_all_files(file_path)
        # save all mat files to list 
        if re.search(r"\d+.mat", file_path):
            all_samples_path.append(file_path)
        elif re.search(r"[a-z]+.mat", file_path):
            labels_path = file_path
    
    return all_samples_path, labels_path

def load_data4classification(samples_path, labels_path, sessions=9):
    '''
    @Description: load the SEED data for classification task
    @param: 
        list_path: list, including all mat files' path
        sessions: the number of training sessions
    '''
    # load the label data
    label = scio.loadmat(labels_path)
    for key, val in label.items():
        if key == "label":
            label  = val + 1
    
    train_sample = None
    train_label = None
    test_sample = None
    test_label = None
    for path in samples_path:
        # load the sample data
        sample = scio.loadmat(path, verify_compressed_data_integrity=False)
        flag = 0
        for key, val in sample.items():
            if key.startswith("de_LDS"):
                # divide the sample data into training and testing data set
                if flag < sessions:
                    if train_sample is None:
                        train_sample = val
                        train_label = np.full((val.shape[1], 1), label[0, flag])
                    else:
                        train_sample = np.concatenate((train_sample, val), axis=1)
                        train_label = np.concatenate((train_label, np.full((val.shape[1], 1), label[0, flag])), axis=0)
                else:
                    if test_sample is None:
                        test_sample = val
                        test_label = np.full((val.shape[1], 1), label[0, flag])
                    else:
                        test_sample = np.concatenate((test_sample, val), axis=1)
                        test_label = np.concatenate((test_label, np.full((val.shape[1], 1), label[0, flag])), axis=0)
                flag += 1

    return train_sample, train_label, test_sample, test_label

def load_data4TL(samples_path, labels_path):
    '''
    @Description: load the SEED data for transfer Learning task
    @param: 
        list_path: list, including all mat files' path
        domains: the number of source domain
    '''
    # load the label
    label = scio.loadmat(labels_path)
    for key, val in label.items():
        if key == "label":
            label = val + 1
    # load the source domain features and the corresponding labels
    sample_feature = None
    sample_label = None
    sample = scio.loadmat(samples_path, verify_compressed_data_integrity=False)
    flag = 0
    for key, val in sample.items():
        if key.startswith("de_LDS"):
            if sample_feature is None and sample_label is None:
                sample_feature = val
                sample_label = np.full((val.shape[1], 1), label[0, flag])
            else:
                sample_feature = np.concatenate((sample_feature, val), axis=1)
                sample_label = np.concatenate((sample_label, np.full((val.shape[1], 1), label[0, flag])), axis=0)
            flag += 1
    
    return sample_feature, sample_label


if __name__ == "__main__":
    # get the path of data directory
    parent_path = os.path.abspath("/home/EGG_Data")
    data_path = os.path.abspath(os.path.join(parent_path, "SEED", "ExtractedFeatures"))
    feature_path, label_path = get_all_files(data_path)
    
    train_sample, train_label, test_sample, test_label = load_data4classification(feature_path, label_path)
    print("train sample shape:{}, test sample shape:{}".format(np.shape(train_sample), np.shape(test_sample)))
    print("train label shape:{}, test label shape:{}".format(np.shape(train_label), np.shape(test_label)))

    print("=====================After reshaping======================")
    train_sample = np.swapaxes(train_sample, 0, 1).reshape(-1, 310)
    test_sample = np.swapaxes(test_sample, 0, 1).reshape(-1, 310)
    print("train sample shape:{}, test sample shape:{}".format(np.shape(train_sample), np.shape(test_sample)))
    print("train label shape:{}, test label shape:{}".format(np.shape(train_label), np.shape(test_label)))
    """
    print()
    flag = 0
    for path in feature_path:
        feature, label= load_data4TL(path, label_path)
        print("the path:{}, The feature shape:{}, the label shape:{}".format(path, feature.shape, label.shape))
    """
    
    