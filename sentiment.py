import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


stemmer = PorterStemmer()

main_directory = "aclImdb/train"
pos_rew = os.path.join(main_directory, "pos")
neg_rew = os.path.join(main_directory, "neg")
txt_files = [f for f in os.listdir(pos_rew) if f.endswith(".txt")]
txt_files_neg = [f for f in os.listdir(neg_rew) if f.endswith(".txt")]

# Create an empty list to store the text data
data = []

for txt_file in txt_files:
    file_path = os.path.join(pos_rew, txt_file)
    with open(file_path, "r") as file:
        content = file.read().casefold()
    # Append the content to the list
    data.append(content)

# Create a DataFrame from the list of text data
df_pos = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df_pos["Label"] = "positive"

# Print the first few rows of the DataFrame to verify
#print(df.head())

for txt_file in txt_files_neg:
    file_path = os.path.join(neg_rew, txt_file)
    with open(file_path, "r") as file:
        content = file.read().casefold()
    # Append the content to the list
    data.append(content)

# Create a DataFrame from the updated list of text data
df_neg = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df_neg["Label"] = "negative"

# Concatenate the two DataFrames
df = pd.concat([df_pos, df_neg])
# Print the first few rows of the DataFrame to verify
#print('here',df)


#Tokenisation
nltk.download('punkt')
df['tokens'] = df['Text'].apply(nltk.word_tokenize)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#stopwords
nltk.download('stopwords')
stopwords.words('english')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Apply the remove_stopwords function to the 'tokens' column
df['tokens'] = df['tokens'].apply(remove_stopwords)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#df['tokens']=stemmer.stem(df['tokens'])
#print(df.head())

def stemming_words(tokens):
    new_tokens=[stemmer.stem(token) for token in tokens]
    return new_tokens

df['tokens']=df['tokens'].apply(stemming_words)
#print(df.head())

#add padding to the end of each review
# Find the maximum length of reviews in the DataFrame
max_length = df['tokens'].apply(len).max()

# Define the padding token
padding_token = '<PAD>'

# Define a function to pad tokens in each review
def pad_tokens(tokens):
    padded_tokens = tokens + [padding_token] * (max_length - len(tokens))
    return padded_tokens

# Apply the pad_tokens function to the 'tokens' column
df['tokens'] = df['tokens'].apply(pad_tokens)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#Word2vec Training - skipgram, hierarchical softmax and vector dimension as 300
from gensim.models import Word2Vec

# Extract the tokenized reviews as a list of lists
tokenized_reviews = df['tokens'].tolist()

# Train the Word2Vec model with skip-gram and hierarchical softmax
# You can adjust the parameters, including the vector dimension, window size, etc.
vector_dimension = 300  # Specify the vector dimension
word2Vec_model = Word2Vec(tokenized_reviews, sg=1, hs=1, vector_size=vector_dimension, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model for later use if needed
word2Vec_model.save("word2vec_model")

# Now, you can use the trained Word2Vec model to obtain word vectors
# For example, to get the vector for a specific word:
word_vector = word2Vec_model.wv['career']

# You can also find similar words to a given word:
similar_words = word2Vec_model.wv.most_similar('career', topn=5)

# Print the word vector and similar words for demonstration
#print("Vector for 'example_word':", word_vector)
#print("Similar words to 'example_word':", similar_words)


#LSTM Training For LSTM, a dropout value set to 0.2, with learning rate value set to 0.001, and with average pooling
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Assuming you have already trained the Word2Vec model and loaded it
word2Vec_model = Word2Vec.load("word2vec_model")

# Convert tokenized reviews to word embeddings
word_vectors = []
for tokens in tokenized_reviews:
    vectors = [word2Vec_model.wv[word] for word in tokens]
    word_vectors.append(vectors)

# Pad the sequences to a fixed length
max_sequence_length = max_length  # Use the maximum length from previous processing
padded_word_vectors = pad_sequences(word_vectors, maxlen=max_sequence_length, padding='post', dtype='float32')

# Create labels for sentiment (assuming 'positive' is 1 and 'negative' is 0)
labels = (df['Label'] == 'positive').astype(int)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_word_vectors, labels, test_size=0.2, random_state=42)

# Build the LSTM model
# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(max_sequence_length, vector_dimension)))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.2))  # Add dropout with rate 0.2
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a custom learning rate
custom_optimizer = Adam(learning_rate=0.001)  # Set the learning rate to 0.001
model.compile(loss='binary_crossentropy', optimizer=custom_optimizer, metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()


# Train the model
epochs = 20  # You can adjust the number of epochs
batch_size = 64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy*100:.2f}%")




main_directory_test = "aclImdb/test"
pos_rew_test = os.path.join(main_directory, "pos")
neg_rew_test = os.path.join(main_directory, "neg")
txt_files_test = [f for f in os.listdir(pos_rew) if f.endswith(".txt")]
txt_files_neg_test = [f for f in os.listdir(neg_rew) if f.endswith(".txt")]

# Create an empty list to store the text data
data_test = []

for txt_file in txt_files_test:
    file_path = os.path.join(pos_rew, txt_file)
    with open(file_path, "r") as file:
        content = file.read().casefold()
    # Append the content to the list
    data_test.append(content)

# Create a DataFrame from the list of text data
df_pos_test = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df_pos_test["Label"] = "positive"

# Print the first few rows of the DataFrame to verify
#print(df.head())

for txt_file in txt_files_neg_test:
    file_path = os.path.join(neg_rew, txt_file)
    with open(file_path, "r") as file:
        content = file.read().casefold()
    # Append the content to the list
    data_test.append(content)

# Create a DataFrame from the updated list of text data
df_neg_test = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df_neg_test["Label"] = "negative"

# Concatenate the two DataFrames
df_test = pd.concat([df_pos, df_neg])
# Print the first few rows of the DataFrame to verify
#print('here',df)


#Tokenisation
nltk.download('punkt')
df_test['tokens'] = df_test['Text'].apply(nltk.word_tokenize)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#stopwords
nltk.download('stopwords')
stopwords.words('english')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

# Apply the remove_stopwords function to the 'tokens' column
df_test['tokens'] = df_test['tokens'].apply(remove_stopwords)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#df['tokens']=stemmer.stem(df['tokens'])
#print(df.head())

def stemming_words(tokens):
    new_tokens=[stemmer.stem(token) for token in tokens]
    return new_tokens

df_test['tokens']=df_test['tokens'].apply(stemming_words)
#print(df.head())

#add padding to the end of each review
# Find the maximum length of reviews in the DataFrame
max_length = df_test['tokens'].apply(len).max()

# Define the padding token
padding_token = '<PAD>'

# Define a function to pad tokens in each review
def pad_tokens(tokens):
    padded_tokens = tokens + [padding_token] * (max_length - len(tokens))
    return padded_tokens

# Apply the pad_tokens function to the 'tokens' column
df_test['tokens'] = df_test['tokens'].apply(pad_tokens)

# Print the first few rows of the DataFrame to verify
#print(df.head())

#Word2vec Training - skipgram, hierarchical softmax and vector dimension as 300
from gensim.models import Word2Vec

# Extract the tokenized reviews as a list of lists
tokenized_reviews_test = df_test['tokens'].tolist()

# Train the Word2Vec model with skip-gram and hierarchical softmax
# You can adjust the parameters, including the vector dimension, window size, etc.
vector_dimension = 300  # Specify the vector dimension
word2Vec_model_test = Word2Vec(tokenized_reviews_test, sg=1, hs=1, vector_size=vector_dimension, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model for later use if needed
word2Vec_model_test.save("word2vec_model_test")

# Now, you can use the trained Word2Vec model to obtain word vectors
# For example, to get the vector for a specific word:
word_vector_test = word2Vec_model_test.wv['career']

# You can also find similar words to a given word:
similar_words = word2Vec_model_test.wv.most_similar('career', topn=5)

# Print the word vector and similar words for demonstration
#print("Vector for 'example_word':", word_vector)
#print("Similar words to 'example_word':", similar_words)

word_vectors_test = []
for tokens in tokenized_reviews_test:
    vectors = [word2Vec_model_test.wv[word] for word in tokens]
    word_vectors_test.append(vectors)

# Pad the sequences to a fixed length (if not done already)
max_sequence_length = max_length  # Use the maximum length from training data preprocessing
padded_word_vectors_test = pad_sequences(word_vectors_test, maxlen=max_sequence_length, padding='post', dtype='float32')

# Predict labels for the test data
y_pred = model.predict(padded_word_vectors_test)
# Assuming you've loaded and preprocessed your test data
# Load the test data and preprocess it, similar to the training data
# Then, define the true labels for the test data
test_labels = (df_test['Label'] == 'positive').astype(int)

# Convert probability scores to binary labels (e.g., if the predicted probability is >= 0.5, consider it positive)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate the model's performance using test labels
accuracy = accuracy_score(test_labels, y_pred_binary)
report = classification_report(test_labels, y_pred_binary)
conf_matrix = confusion_matrix(test_labels, y_pred_binary)

# Print or display the evaluation results
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)