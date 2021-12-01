
# STL
import re
import string
from json import dumps
from email.parser import BytesParser
from email.policy import default

# PDM
import numpy as np
from flask import Response, Blueprint, jsonify, request, redirect, send_file
from tensorflow import keras

bp = Blueprint("routes", __name__)



def read_email(file):
    """
    Raed the raw content from an email.

    Params:
     - path (str): the file path for the email to read.
    
    Returns:
     - content (str): raw content of an email.
    """
    
    msg = BytesParser(policy=default).parse(file)
    for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload()  # raw text
        


def clean_data(data):
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
    printable = set(string.printable)
    data = ''.join(filter(lambda x: x in printable, data))
    return data


@bp.route("/predict", methods=["POST"])
def predict():

    vocab_size = 50000
    max_words = 2000
    m = keras.models.load_model("model")
    assert m, "Model couldn't be loaded"

    results = dict()
    files = request.files.getlist("file")
    for _f in files:
        data = read_email(_f)
        data = clean_data(data)
        print(f"{data}\n\n")
        x_predict = [data]
        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(x_predict)

        new_features = keras.preprocessing.sequence.pad_sequences(
            np.array(tokenizer.texts_to_sequences(x_predict), dtype=object),  # basically one-hot-encoding
            maxlen=max_words
        )
        prediction = m.predict(new_features)[0][0] * 100
        print(prediction)
        res = "Spam" if prediction >= 1 else "Ham"
        results[_f.filename] = res

    return dumps(results)


@bp.route("/", methods=["GET"])
@bp.route("/<path>", methods=["GET"])
def index(path="") -> Response:
    print("here1")
    return send_file("./index.html")
