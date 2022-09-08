from mimetypes import init
import numpy as np
import tensorflow as tf
from tensorflow import keras

class AutocompleteUtils():
    def __init__(self) -> None:
        book_url = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
        filepath = keras.utils.get_file("nietzsche.txt", book_url)
        with open(filepath) as f:
            nietzsche_text = f.read()
        self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(nietzsche_text)
        self.max_id = len(self.tokenizer.word_index)
        self.dataset_size = self.tokenizer.document_count

    def preprocess(self, texts):
        X = np.array(self.tokenizer.texts_to_sequences(texts)) - 1
        return tf.one_hot(X, self.max_id)

    def next_chart(self, model, text, temperature=1):
        X_new = self.preprocess([text])
        Y_proba = model.predict(X_new)[0, -1:, :]
        rescaled_logits = tf.math.log(Y_proba) / temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
        return self.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def complete_text(self, model, text, n_charts=50, temperature=1):
        for _ in range(n_charts):
            text += self.next_chart(model, text, temperature)
        return text