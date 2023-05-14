import pandas as pd
import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# tokenize the queries
import tensorflow as tf
import os

# Get the current directory path
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the file
file_path = os.path.join(current_dir, 'helper_files', 'totalclassifierdata.txt')
with open(file_path, 'r') as f:
    lines = f.readlines()
    data = {'query': [], 'label': []}
    for line in lines:
        line = line.strip().split('\t')
        if len(line) >= 2:
            data['query'].append(line[0])
            data['label'].append(line[1])
        else:
            continue
    dataset = pd.DataFrame(data)
# extract the input queries and their corresponding labels
queries = dataset['query']
labels = dataset['label']

# tokenize the queries
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(queries)
vocab_size = len(tokenizer.word_index) + 1
file_path_model = os.path.join(current_dir, 'helper_files', 'classifier-NN')

model = tf.keras.models.load_model(file_path_model)
sequences = tokenizer.texts_to_sequences(queries)
max_len = max([len(seq) for seq in sequences])

# classify a new input query
def classify(new_query):
    new_query_sequence = tokenizer.texts_to_sequences([new_query])
    new_query_padded = tf.keras.preprocessing.sequence.pad_sequences(new_query_sequence, maxlen=max_len, padding='post')
    prediction = np.argmax(model.predict(new_query_padded), axis=-1)
    if prediction == 1:
        print("The input query is a topical query.")
        return 2
    else:
        print("The input query is a chitchat query.")
        return 1