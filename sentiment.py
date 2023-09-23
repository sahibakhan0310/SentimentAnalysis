import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

main_directory = ".\\aclImdb\\train"
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

#Word2vec Training
from gensim.models import Word2Vec

# Extract the tokenized reviews as a list of lists
tokenized_reviews = df['tokens'].tolist()

# Train the Word2Vec model with skip-gram and hierarchical softmax
# You can adjust the parameters, including the vector dimension, window size, etc.
vector_dimension = 300  # Specify the vector dimension
model = Word2Vec(tokenized_reviews, sg=1, hs=1, vector_size=vector_dimension, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model for later use if needed
model.save("word2vec_model")

# Now, you can use the trained Word2Vec model to obtain word vectors
# For example, to get the vector for a specific word:
word_vector = model.wv['career']

# You can also find similar words to a given word:
similar_words = model.wv.most_similar('career', topn=5)

# Print the word vector and similar words for demonstration
#print("Vector for 'example_word':", word_vector)
#print("Similar words to 'example_word':", similar_words)


#LSTM Training



