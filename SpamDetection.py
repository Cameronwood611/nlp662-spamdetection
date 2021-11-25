# STL
import os
import re
import glob
import email

# PDM
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split


def all_datasets_exist() -> bool:
    """

    Function that checks for all the expected spam datasets to be unpacked. There
    are existing tar files in test-data but they need to be unpacked manually.

    Params:
        - None

    Returns:
        - True or False

    """

    expected_dirs = ["easy_ham", "easy_ham_2", "hard_ham", "spam", "spam_2"]
    folder = "./test-data/"

    for dir_name in expected_dirs:
        path = folder + dir_name
        if not os.path.exists(path):
            return False
    return True


# Pull content of email
def get_email_content(email_paths):
    content = []
    for path in email_paths:
        file = open(path, encoding="latin1")
        msg = email.message_from_file(file)
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                content.append(part.get_payload())  # raw text
    return content


def remove_null(datasets, labels):
    """
    Sometimes email content is empty when we receive it. We can't remove it as we
    read the email becuase we have a matching y_label unpacked already and we need
    to remove them at the same time.

    Params:
     - datasets (list): Raw emails to be evaluated.
     - labels: (list): 1/0 y_label of the emails from dataset.
    
    Returns:
     - (datasets, labels): removed of any null content
    """
    not_null_idx = [i for i, o in enumerate(datasets) if o is not None]
    return np.array(datasets)[not_null_idx], np.array(labels)[not_null_idx]


def prepare_datasets():
    """
    Start of the SpamDetection pipeline which involves actually unpacking and splitting
    the datasets into train and test data. You'll need to make sure to unpack the tar files
    inside the test-data directory (see the function `all_datasets_exist` for more detail).

    Params:
     - None

     Returns:
     - 4-tuple of train/test datasets with y labels.
    """
    assert (
        all_datasets_exist()
    ), "Need to unpack tar files to get datasets in test-data directory"

    # gather all file names unpacked
    path = "./test-data/"
    ham_files = [
        glob.glob(path + "easy_ham/*"),
        glob.glob(path + "easy_ham_2/*"),
        glob.glob(path + "hard_ham/*"),
    ]
    spam_files = [glob.glob(path + "spam/*"), glob.glob(path + "spam_2/*")]

    ham_sample = np.array([train_test_split(o) for o in ham_files], dtype=object)

    ham_train = np.array([])
    ham_test = np.array([])
    for o in ham_sample:
        ham_train = np.concatenate((ham_train, o[0]), axis=0)
        ham_test = np.concatenate((ham_test, o[1]), axis=0)

    spam_sample = np.array([train_test_split(o) for o in spam_files])

    spam_train = np.array([])
    spam_test = np.array([])
    for o in spam_sample:
        spam_train = np.concatenate((spam_train, o[0]), axis=0)
        spam_test = np.concatenate((spam_test, o[1]), axis=0)

    # attach labels to data (0 - ham, 1 - spam)
    ham_train_label = [0] * ham_train.shape[0]  # type: ignore
    spam_train_label = [1] * spam_train.shape[0]  # type: ignore
    ham_test_label = [0] * ham_test.shape[0]  # type: ignore
    spam_test_label = [1] * spam_test.shape[0]  # type: ignore

    x_test = np.concatenate((ham_test, spam_test))
    y_test = np.concatenate((ham_test_label, spam_test_label))
    x_train = np.concatenate((ham_train, spam_train))
    y_train = np.concatenate((ham_train_label, spam_train_label))

    # shuffle
    train_shuffle_index = np.random.permutation(np.arange(0, x_train.shape[0]))  # type: ignore
    test_shuffle_index = np.random.permutation(np.arange(0, x_test.shape[0]))  # type:ignore

    x_train = x_train[train_shuffle_index]
    y_train = y_train[train_shuffle_index]

    x_test = x_test[test_shuffle_index]
    y_test = y_test[test_shuffle_index]

    # email content for training
    x_train = get_email_content(x_train)
    x_test = get_email_content(x_test)

    x_train, y_train = remove_null(x_train, y_train)
    x_test, y_test = remove_null(x_test, y_test)


def main():
    prepare_datasets()


main()
