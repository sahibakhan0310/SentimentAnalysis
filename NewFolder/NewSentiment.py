import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from itertools import islice
from nltk.stem.porter import PorterStemmer

import unicodedata

def load_data():
    main_directory = "aclImdb/train"
    pos_rew = os.path.join(main_directory, "pos")
    neg_rew = os.path.join(main_directory, "neg")
    txt_files = [f for f in os.listdir(pos_rew) if f.endswith(".txt")]
    txt_files_neg = [f for f in os.listdir(neg_rew) if f.endswith(".txt")]

    data = []

    for txt_file in txt_files:
        file_path = os.path.join(pos_rew, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
            content = ''.join(c for c in content if unicodedata.category(c) != 'So')
            content = content.casefold()
        data.append(content)

    df_pos = pd.DataFrame(data, columns=["Text"])
    df_pos["Label"] = "positive"

    for txt_file in txt_files_neg:
        file_path = os.path.join(neg_rew, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
            content = ''.join(c for c in content if unicodedata.category(c) != 'So')
            content = content.casefold()
        data.append(content)

    df_neg = pd.DataFrame(data, columns=["Text"])
    df_neg["Label"] = "negative"

    df = pd.concat([df_pos, df_neg])

    return df

def tokenize_data(df):
    nltk.download('punkt')
    df['tokens'] = df['Text'].apply(nltk.word_tokenize)
    return df

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def preprocess_data(df):
    nltk.download('stopwords')
    df['tokens'] = df['tokens'].apply(remove_stopwords)

    def stemming_words(tokens):
        new_tokens = [stemmer.stem(token) for token in tokens]
        return new_tokens

    df['tokens'] = df['tokens'].apply(stemming_words)

    max_length = df['tokens'].apply(len).max()
    padding_token = '<PAD>'

    def pad_tokens(tokens):
        padded_tokens = tokens + [padding_token] * (max_length - len(tokens))
        return padded_tokens

    df['tokens'] = df['tokens'].apply(pad_tokens)

    df.to_csv('padded_tokens.csv', index=False)
    return df

def train_word2vec_model(df):
    tokenized_reviews = df['tokens'].tolist()
    word2Vec_model = Word2Vec(tokenized_reviews, sg=1, hs=1, vector_size=vector_dimension, window=5, min_count=1, workers=8)
    word2Vec_model.save("word2vec_model")
    return word2Vec_model

def load_word2vec_model():
    return Word2Vec.load("word2vec_model")

def convert_to_word_embeddings(df, word2Vec_model):
    review_vectors = []
    labels = []

    for i, tokens in enumerate(df['tokens']):
        vectors = [word2Vec_model.wv[word] for word in tokens if word in word2Vec_model.wv.key_to_index]
        if vectors:
            review_vector = np.mean(vectors, axis=0)
            review_vectors.append(review_vector)
            label = int(df['Label'].iloc[i] == 'positive')
            labels.append(label)

    sequence_lengths = [len(seq) for seq in review_vectors]
    sequence_lengths.sort()
    max_sequence_length = sequence_lengths[int(len(sequence_lengths) * 0.95)]

    padded_word_vectors = pad_sequences(review_vectors, maxlen=max_sequence_length, padding='post', dtype='float32')
    labels = (df['Label'] == 'positive').astype(int)

    return padded_word_vectors, labels

def split_data(padded_word_vectors, labels):
    X_train, X_val, y_train, y_val = train_test_split(padded_word_vectors, labels, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    return X_train, X_val, y_train, y_val

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(1, 300)))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])
    model.summary()
    model.save('lstm_model.h5')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    classification_rep = classification_report(y_val, y_pred)
    confusion_mat = confusion_matrix(y_val, y_pred)

    print("Accuracy: ", accuracy)
    print("Classification Report: ", classification_rep)
    print("Confusion Matrix: ", confusion_mat)

# Main execution
stemmer = PorterStemmer()
vector_dimension = 300

df = load_data()
df = tokenize_data(df)
df = preprocess_data(df)

word2Vec_model = train_word2vec_model(df)
word2Vec_model = load_word2vec_model()

padded_word_vectors, labels = convert_to_word_embeddings(df, word2Vec_model)

X_train, X_val, y_train, y_val = split_data(padded_word_vectors, labels)

model = build_lstm_model()
history = train_model(model, X_train, y_train, X_val, y_val)

evaluate_model(model, X_val, y_val)
