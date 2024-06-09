# File: protein_sequence_classification.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    sequences = data['sequence'].values
    labels = data['label'].values
    return sequences, labels

def preprocess_data(sequences, labels, max_len=100):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return sequences, labels, tokenizer, label_encoder

# Build RNN model
def build_model(input_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_file = 'path/to/protein_data.csv'
    sequences, labels = load_data(data_file)
    sequences, labels, tokenizer, label_encoder = preprocess_data(sequences, labels)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    model = build_model(input_length=X_train.shape[1], vocab_size=len(tokenizer.word_index) + 1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    # Save the model
    model.save('protein_sequence_classification_model.h5')

    # Save tokenizer and label encoder
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
