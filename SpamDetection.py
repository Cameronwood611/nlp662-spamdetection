# STL
import os
import re
import glob
import email
import string

# PDM
import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense, Input, Dropout, Embedding, Bidirectional
from keras.models import Model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from  matplotlib import pyplot as plt
import seaborn as sns

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


def read_email(path):
    file = open(path, encoding="latin1")
    try:
        msg = email.message_from_file(file)
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload()  # raw text
    except Exception as e:
        print(e)
    

def get_email_content(email_paths):
    content = [read_email(path) for path in email_paths]
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

    return x_train, y_train, x_test, y_test

def preprocess_clean(data):
    """
    Do some preprocess cleaning with the data from emails. We want to do some
    normalization so there is no bias for certain anomalies within the data.

    Params:
     - data (str): The raw content of an email

    Returns:
     - data (str): normalized and clean data
    """

    data = data.replace("\n", "").lower().strip()
    data = re.sub(r"http\S+", "", data)  # no hyperlink
    data = re.sub(r'\d+', '', data) # no numbers
    data = data.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    return data


def build_features(x_train, x_test):
    max_feature = 50000  # how many unique words
    max_len = 2000 # max number of words

    tokenizer = Tokenizer(num_words=max_feature)

    tokenizer.fit_on_texts(x_train)
    x_train_features = np.array(tokenizer.texts_to_sequences(x_train), dtype=object)
    x_test_features = np.array(tokenizer.texts_to_sequences(x_test), dtype=object)

    x_train_features = pad_sequences(x_train_features,maxlen=max_len)
    x_test_features = pad_sequences(x_test_features,maxlen=max_len)

    return x_train_features, x_test_features


def create_model():
    # create the model
    max_feature = 50000  # how many unique words
    max_len = 2000 # max number of words
    embedding_vecor_length = 32

    model = tf.keras.Sequential()
    model.add(Embedding(max_feature, embedding_vecor_length, input_length=max_len))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def printPlot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    plt.show()
    
def printConfusionMatrix(y_test, y_predict):
    cf_matrix = confusion_matrix(y_test, y_predict)
    
    ax = plt.subplot()
    sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt=''); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); ax.yaxis.set_ticklabels(['Not Spam', 'Spam']);
    
def printScores(y_test, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_predict)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_predict)))
    print("F1 Score: {:.2f}%".format(100 * f1_score(y_test,y_predict)))

def main():
    x_train, y_train, x_test, y_test = prepare_datasets()
    x_train = [preprocess_clean(o) for o in x_train]
    x_test = [preprocess_clean(o) for o in x_test]

    x_train_features, x_test_features = build_features(x_train, x_test)

    model = create_model()
    history = model.fit(
        x_train_features, y_train, batch_size=512, epochs=20, validation_data=(x_test_features, y_test))
    
    # Predict scores
    y_predict  = [1 if o>0.5 else 0 for o in model.predict(x_test_features)]
    
    printPlot(history)
    printConfusionMatrix(y_test, y_predict)
    printScores(y_test, y_predict)

main()
