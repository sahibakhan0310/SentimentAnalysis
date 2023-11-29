import os
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, GlobalAveragePooling1D, Bidirectional
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences

# Download and load pre-trained GloVe embeddings
glove_file = '/Users/sahibakhan/resources/ML/project/SentimentAnalysis/glove/glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.word2vec'
glove2word2vec(glove_file, word2vec_output_file)
word2Vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

def load_data():
    main_directory = "/Users/sahibakhan/resources/ML/project/aclImdb/train"
    pos_rew = os.path.join(main_directory, "pos")
    neg_rew = os.path.join(main_directory, "neg")

    data = []

    for txt_file in os.listdir(pos_rew):
        file_path = os.path.join(pos_rew, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read().casefold()
        data.append(content)

    df_pos = pd.DataFrame(data, columns=["Text"])
    df_pos["Label"] = "positive"

    data = []

    for txt_file in os.listdir(neg_rew):
        file_path = os.path.join(neg_rew, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read().casefold()
        data.append(content)

    df_neg = pd.DataFrame(data, columns=["Text"])
    df_neg["Label"] = "negative"

    df = pd.concat([df_pos, df_neg])

    return df

def preprocess_data(df):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    df['tokens'] = df['Text'].apply(lambda text: [stemmer.stem(word) for word in text.split() if word.lower() not in stop_words])

    max_length = df['tokens'].apply(len).max()
    padding_token = '<PAD>'

    df['tokens'] = df['tokens'].apply(lambda tokens: tokens + [padding_token] * (max_length - len(tokens)))

    df.to_csv('padded_tokens.csv', index=False)
    return df

def convert_to_word_embeddings(df):
    review_vectors = []
    labels = []

    for i, tokens in enumerate(df['tokens']):
        vectors = [word2Vec_model[word] for word in tokens if word in word2Vec_model]
        if vectors:
            review_vector = np.mean(vectors, axis=0)
            review_vectors.append(review_vector)
            label = int(df['Label'].iloc[i] == 'positive')
            labels.append(label)

    max_sequence_length = max(len(seq) for seq in review_vectors)

    padded_word_vectors = pad_sequences(review_vectors, maxlen=max_sequence_length, padding='post', dtype='float32')
    labels = (df['Label'] == 'positive').astype(int)

    return padded_word_vectors, labels

def split_data(padded_word_vectors, labels):
    X_train, X_val, y_train, y_val = train_test_split(padded_word_vectors, labels, test_size=0.2, random_state=42)
    # Reshape to (batch_size, sequence_length, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    return X_train, X_val, y_train, y_val



def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(200, return_sequences=True), input_shape=input_shape))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])
    model.summary()
    return model


def build_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_logistic_regression_model():
    return LogisticRegression(random_state=42)

def build_stacking_model(base_models, meta_model, input_shape):
    base_models_with_input_shape = [(name, build_model_fn(input_shape)) for name, build_model_fn in base_models]
    stacking_model = StackingClassifier(estimators=base_models_with_input_shape, final_estimator=meta_model, cv=5)
    return stacking_model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    model.fit(X_train.squeeze(), y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val.squeeze())
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    classification_rep = classification_report(y_val, y_pred)
    confusion_mat = confusion_matrix(y_val, y_pred)

    print("Accuracy: ", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)

# Main execution
df = load_data()
df = preprocess_data(df)

padded_word_vectors, labels = convert_to_word_embeddings(df)

X_train, X_val, y_train, y_val = split_data(padded_word_vectors, labels)

# Build and train stacking model
input_shape = (X_train.shape[1], 1)
stacking_model = build_stacking_model(base_models, logistic_model, input_shape)
train_model(stacking_model, X_train, y_train, X_val, y_val)

# Evaluate the stacking model
evaluate_model(stacking_model, X_val, y_val)
