# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk import word_tokenize
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# loading the data
data = pd.read_csv("personality data.csv")
# print("Data: \n", data)
# print(data.shape)

# Exploring the data

# getting data info
# data.info()

# describing the data
# print("\n Data Description: \n", data.describe())

# counting the total particular types in the data
type_count = data['type'].value_counts()
# print("\n Type Count: \n", type_count)

# plotting the personalities over their value counts
plt.figure(figsize=(8, 6), facecolor='skyblue')

plt.hlines(y=list(range(16)), xmin=0, xmax=type_count, colors='blue', )

plt.plot(type_count, list(range(16)), 'o')
plt.yticks(list(range(16)), type_count.index)

plt.ylabel('Personality Types')
plt.xlabel('Value Count')

# plt.show()

# separating the posts from ||| delimiter and counting total posts per id
data['separated_posts'] = data['posts'].apply(lambda x: x.strip().split('|||'))
data['num_post'] = data['separated_posts'].apply(lambda x: len(x))

# print("\n Data: \n", data.head())

# unique values from num_post
# print(data['num_post'].unique())

# taking the first data of separated_post and converting to list
# string = list(data['separated_posts'][1])
# len(string) = 50
# print(string)

# grouping the num_post as per the types
num_post_data = data.groupby('type')['num_post'].apply(list).reset_index()
# print(num_post_data)

# plotting the number of posts per personality
plt.title('Number of Posts per Personality')
plt.figure(figsize=(8, 6), facecolor='pink')

sns.violinplot(x=data.type, y=data.num_post)
# sns.barplot(x=data.type, y=data.num_post)

plt.xlabel('Personality Types')
plt.ylabel('Number of Posts')


# plt.show()


# cleaning the data
def clean_data(text):
    # removing the links and change case to lower case
    result = re.sub(r'https?://\S+|www\.\S+', '', text).lower()

    # removing the numbers
    result = re.sub(r'[0-9]+', '', result)

    # removing punctuations
    result = re.sub(r'[^\w\s]', '', result)

    # removing extra whitespaces
    result = " ".join(result.split())

    return result


'''def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)'''

# list to string -> separated_posts
data['string_posts'] = ['\\\\'.join(map(str, x)) for x in data['separated_posts']]
# print(data['string_posts'].head(1))

data['clean_posts'] = data['string_posts'].apply(clean_data)
# data['string_posts'] = data['string_posts'].apply(remove_punct)
# print(data['string_posts'])

# word tokenization
print('Tokenizing the data.')
data['tokenized_posts'] = data['clean_posts'].apply(lambda x: word_tokenize(x))
# print(data['string_posts'])


# checking the spellings in data using spellcheck
# creating a spell_check function
'''print('Spell Check \n')
def spell_check(text):
    result = []
    spell = SpellChecker()
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word)

    return result


data['string_posts'] = data['string_posts'].apply(spell_check)
print(data['string_posts'])'''

# removing stop words
# print(stopwords.words('english'))

stop_words = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj', 'infjs', 'entps', 'intps', 'intjs', 'entjs', 'enfjs', 'infps', 'enfps', 'isfps', 'istps', 'isfjs', 'istjs', 'estps', 'esfps', 'estjs', 'esfjs']

en_stopwords = stopwords.words('english')
en_stopwords.extend(stop_words)
# print(en_stopwords)

print('Remove Stop Words')


def remove_stopwords(text):
    result = []

    for token in text:
        if token not in en_stopwords:
            result.append(token)

    return result


# applying remove_stopwords over dataset.
data['tokenized_posts'] = data['tokenized_posts'].apply(remove_stopwords)
# print(data['string_posts'])

print('Normalizing the data')


# Normalizing the data
def lemmatize_token(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# applying the lemmatize function over dataset
data['normalized_posts'] = data['tokenized_posts'].apply(lemmatize_token)
# print(data['string_posts'])

# list to string
data['cleaned_posts'] = data['normalized_posts'].apply(lambda x: ' '.join(x))
# print(data['cleaned_posts'])

# Vectorizing the data and removing some extra stop words
print('Vectorizing data')

val = data['cleaned_posts'].values.reshape(1, -1).tolist()[0]
# print(val)

vectorizer = CountVectorizer()
# vectorizer.fit(val)
x_cnt = vectorizer.fit_transform(val)
# print(x_cnt)

# Transform the count matrix to a tf-idf representation
tfizer = TfidfTransformer()
# tfizer.fit(x_cnt)
X = tfizer.fit_transform(x_cnt).toarray()
print(X.shape)


