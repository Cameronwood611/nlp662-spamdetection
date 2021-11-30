# PDM
import numpy as np
from tensorflow import keras

m = keras.models.load_model("model")
assert m, "Model couldn't be loaded"

vocab_size = 50000
max_words = 2000
x_new = ["some random sentence that really should not be classified as spam but well see"]
print(x_new)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_new)

new_features = keras.preprocessing.sequence.pad_sequences(
    np.array(tokenizer.texts_to_sequences(x_new), dtype=object),  # basically one-hot-encoding
    maxlen=max_words
)
print(new_features.shape)
print(m.predict(new_features))
