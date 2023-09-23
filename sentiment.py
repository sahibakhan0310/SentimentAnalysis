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
df = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df["Label"] = "positive"

# Print the first few rows of the DataFrame to verify
#print(df.head())

for txt_file in txt_files_neg:
    file_path = os.path.join(neg_rew, txt_file)
    with open(file_path, "r") as file:
        content = file.read().casefold()
    # Append the content to the list
    data.append(content)

# Create a DataFrame from the updated list of text data
df = pd.DataFrame(data, columns=["Text"])

# Process the content of the DataFrame as needed
# For example, you can add labels or perform further analysis
df["Label"] = "negative"

# Print the first few rows of the DataFrame to verify
#print(df.head())


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
print(df.head())
