import os
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, GlobalAveragePooling1D, Bidirectional
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Download and load pre-trained GloVe embeddings
glove_file = './glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.word2vec'
glove2word2vec(glove_file, word2vec_output_file)
word2Vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

def load_data(main_directory):
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

def convert_to_word_embeddings(df, max_sequence_length):
    review_vectors = []
    labels = []

    for i, tokens in enumerate(df['tokens']):
        vectors = [word2Vec_model[word] for word in tokens if word in word2Vec_model]
        if vectors:
            review_vector = np.mean(vectors, axis=0)
            if not np.isnan(np.sum(review_vector)):  # Check if the vector is not NaN
                review_vectors.append(review_vector)
                label = int(df['Label'].iloc[i] == 'positive')
                labels.append(label)

    padded_word_vectors = pad_sequences(review_vectors, maxlen=max_sequence_length, padding='post', dtype='float32')
    labels = (df['Label'] == 'positive').astype(int)

    return padded_word_vectors, labels


def split_data(padded_word_vectors, labels):
    X_train, X_val, y_train, y_val = train_test_split(padded_word_vectors, labels, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
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
    model.save('lstm_model.h5')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_val, y_pred)
    classification_rep = classification_report(y_val, y_pred)
    confusion_mat = confusion_matrix(y_val, y_pred)

    print("Accuracy: ", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)


# Main execution
main_directory = "./aclImdb/train"
df = load_data(main_directory)
df = preprocess_data(df)
max_sequence_length = df['tokens'].apply(len).max()
padded_word_vectors, labels = convert_to_word_embeddings(df, max_sequence_length)

X_train, X_val, y_train, y_val = split_data(padded_word_vectors, labels)

input_shape = X_train.shape[1:]
'''
model = build_lstm_model(input_shape)
history = train_model(model, X_train, y_train, X_val, y_val)
'''

from keras.models import load_model
# Load the saved model
model = load_model('lstm_model.h5')

evaluate_model(model, X_val, y_val)

def load_test_data(test_directory):
    pos_test_dir = os.path.join(test_directory, "pos")
    neg_test_dir = os.path.join(test_directory, "neg")

    data = []
    labels = []

    # Load positive reviews
    for txt_file in os.listdir(pos_test_dir):
        file_path = os.path.join(pos_test_dir, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read().casefold()
        data.append(content)
        labels.append(1)  # Positive review

    # Load negative reviews
    for txt_file in os.listdir(neg_test_dir):
        file_path = os.path.join(neg_test_dir, txt_file)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read().casefold()
        data.append(content)
        labels.append(0)  # Negative review

    df_test = pd.DataFrame({'Text': data, 'Label': labels})
    return df_test

def classify_test_data(model, df_test,  max_sequence_length):
    df_test = preprocess_data(df_test)
    X_test, y_test = convert_to_word_embeddings(df_test, max_sequence_length)
    X_test = np.expand_dims(X_test, axis=1)  # Add this line

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)

    return y_test, y_pred

# Load and preprocess the test data
test_directory = "./aclImdb/test"
df_test = load_test_data(test_directory)
df_test = preprocess_data(df_test)

# Convert the test data to word embeddings
X_test, y_test = convert_to_word_embeddings(df_test, max_sequence_length)

# Classify the test data
y_test, y_pred_test = classify_test_data(model, df_test,  max_sequence_length)

print('Test data classification results:#########')
print(f'Classification Report:\n {classification_report(y_test, y_pred_test)}')
print(f'Confusion Matrix:\n {confusion_matrix(y_test, y_pred_test)}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_test)}')