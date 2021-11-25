
# STL
import os
import re
import glob

# PDM
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split


def all_datasets_exist():
    """
    Function that checks for all the expected spam datasets to be unpacked. There
    are existing tar files in test-data but they need to be unpacked manually.

    !tar xvjf 20030228_easy_ham_2.tar.bz2
    !tar xvjf 20030228_easy_ham.tar.bz2
    !tar xvjf 20030228_hard_ham.tar.bz2
    !tar xvjf 20030228_spam.tar.bz2
    !tar xvjf 20050311_spam_2.tar.bz2

    """

    expected_dirs = ["easy_ham", "easy_ham_2", "hard_ham", "spam", "spam_2"]
    folder = "./test-data/"

    for dir_name in expected_dirs:
        path = folder + dir_name
        if not os.path.exists(path):
            return False
    return True


def prepare_datasets():
    assert all_datasets_exist(), "Need to unpack tar files to get datasets in test-data directory"

    path = "./test-data/"
    ham_files = [
        glob.glob(path+'easy_ham/*'),
        glob.glob(path+'easy_ham_2/*'),
        glob.glob(path+'hard_ham/*')
    ]
    spam_files = [
        glob.glob(path+'spam/*'),
        glob.glob(path+'spam_2/*')
    ]

    ham_sample = np.array([train_test_split(o) for o in ham_files], dtype=object)

    ham_train = np.array([])
    ham_test = np.array([])
    for o in ham_sample:
        ham_train = np.concatenate((ham_train, o[0]),axis=0)
        ham_test = np.concatenate((ham_test, o[1]),axis=0)

    spam_sample = np.array([train_test_split(o) for o in spam_files])

    spam_train = np.array([])
    spam_test = np.array([])
    for o in spam_sample:
        spam_train = np.concatenate((spam_train,o[0]),axis=0)
        spam_test = np.concatenate((spam_test,o[1]),axis=0)

    # Attach labels to data (0 - ham, 1 - spam)
    ham_train_label = [0]*ham_train.shape[0]  # type: ignore
    spam_train_label = [1]*spam_train.shape[0] # type: ignore
    ham_test_label = [0]*ham_test.shape[0] # type: ignore
    spam_test_label = [1]*spam_test.shape[0] # type: ignore

    x_test = np.concatenate((ham_test,spam_test))
    y_test = np.concatenate((ham_test_label,spam_test_label))
    x_train = np.concatenate((ham_train,spam_train))
    y_train = np.concatenate((ham_train_label,spam_train_label))



def main():
    prepare_datasets()

main()
