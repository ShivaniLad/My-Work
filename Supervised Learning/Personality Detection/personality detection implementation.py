# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk import word_tokenize
from spellchecker import SpellChecker
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_importance
from pylab import rcParams
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# loading the data
data = pd.read_csv("personality data.csv")
print("Data: \n", data)
print(data.shape)

# Exploring the data

# getting data info
data.info()

# describing the data
print("\n Data Description: \n", data.describe())

# counting the total particular types in the data
type_count = data['type'].value_counts()
print("\n Type Count: \n", type_count)


def close_event():
    # timer calls this function after 3 seconds and closes the window
    plt.close()


# plotting the personalities over their value counts
fig1 = plt.figure(figsize=(8, 6), facecolor='skyblue')

# creating a timer object and setting an interval of 3000 milliseconds
timer = fig1.canvas.new_timer(interval=3000)

timer.add_callback(close_event)

plt.hlines(y=list(range(16)), xmin=0, xmax=type_count, colors='blue', )

plt.plot(type_count, list(range(16)), 'o')
plt.yticks(list(range(16)), type_count.index)

plt.ylabel('Personality Types')
plt.xlabel('Value Count')

timer.start()
plt.show()
timer.stop()

# separating the posts from ||| delimiter and counting total posts per id
data['separated_posts'] = data['posts'].apply(lambda x: x.strip().split('|||'))
data['num_post'] = data['separated_posts'].apply(lambda x: len(x))

print("\n Data: \n", data.head())

# unique values from num_post
print(data['num_post'].unique())

# taking the first data of separated_post and converting to list
# string = list(data['separated_posts'][1])
# len(string) = 50
# print(string)

# grouping the num_post as per the types
num_post_data = data.groupby('type')['num_post'].apply(list).reset_index()
print(num_post_data)

# plotting the number of posts per personality
fig2 = plt.figure(figsize=(8, 6), facecolor='pink')

# creating a timer object and setting an interval of 3000 milliseconds
timer = fig2.canvas.new_timer(interval=3000)

timer.add_callback(close_event)

sns.violinplot(x=data.type, y=data.num_post)
# sns.barplot(corpus=data.type, y=data.num_post)

plt.xlabel('Personality Types')
plt.ylabel('Number of Posts')
plt.title('Number of Posts per Personality')

timer.start()
plt.show()
timer.stop()

# list to string -> separated_posts
print('\n List to string.')
data['string_posts'] = [' '.join(map(str, x)) for x in data['separated_posts']]
print(data['string_posts'].head(1))

print('\n Cleaning the posts.')


# cleaning the data
def clean_data(text):
    # removing the links and change case to lower case
    result = re.sub(r'https?://\S+|www\.\S+', '', text).lower()

    # removing the numbers
    result = re.sub(r'[0-9]+', '', result)

    # removing punctuations
    result = re.sub(r'[^\w\s]|_+', '', result)

    # removing extra whitespaces
    result = " ".join(result.split())

    return result


'''def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)'''

data['clean_posts'] = data['string_posts'].apply(clean_data)
print(data['clean_posts'])

# word tokenization
print('\n Tokenizing the data.')
data['tokenized_posts'] = data['clean_posts'].apply(lambda x: word_tokenize(x))
print(data['tokenized_posts'])

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

stop_words = ['im', 'infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj',
              'estp', 'esfp', 'estj', 'esfj', 'infjs', 'entps', 'intps', 'intjs', 'entjs', 'enfjs', 'infps', 'enfps',
              'isfps', 'istps', 'isfjs', 'istjs', 'estps', 'esfps', 'estjs', 'esfjs']

en_stopwords = stopwords.words('english')
en_stopwords.extend(stop_words)
# print(en_stopwords)

print('\n Removing Stop Words.')


def remove_stopwords(text):
    result = []

    for token in text:
        if token not in en_stopwords:
            result.append(token)

    return result


# applying remove_stopwords over dataset.
data['cleaned_posts'] = data['tokenized_posts'].apply(remove_stopwords)
print(data['cleaned_posts'])

print('\n Normalizing the data.')


# Normalizing the data
def lemmatize_token(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# applying the lemmatize function over dataset
data['normalized_posts'] = data['cleaned_posts'].apply(lemmatize_token)
print(data['normalized_posts'])

# list to string
print('\n List to string.')
data['cleaned_posts'] = data['normalized_posts'].apply(lambda x: ' '.join(x))
print(data['cleaned_posts'])

# featuring types of personalities
'''
We have 4 pair of of personalities and accordingly we have made their different columns.

Favorite world ( Extrovert (E) / Introvert (I) ) (First Letter)
Information ( Sensing (S) / Intuition (N) ) (Second Letter)
Decision ( Thinking (T) , Feeling (F) ) (Third Letter)
Structure ( Judging (J) , Perceiving (P) ) (Fourth Letter)
'''

# print(data['type'].head(10))
# preparing y
data['fav_world'] = data['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
data['info'] = data['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
data['decision'] = data['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
data['structure'] = data['type'].apply(lambda x: 1 if x[3] == 'J' else 0)

# print(data.columns)

# building machine learning algorithm
'''print('\n Building Algorithm. ')

print('Splitting the data.')'''

corpus = data['cleaned_posts'].values.reshape(1, -1).tolist()[0]
# print(corpus)

# vectorizing the xdata using countvectorizer
print('\n Vectorizing the data.')

vectorizer = CountVectorizer(max_features=1500, analyzer="word", max_df=0.8, min_df=0.1)
xcnt = vectorizer.fit_transform(corpus)
# print(xcnt)

# transforming the vectorized to tf-idf form
tfidf = TfidfTransformer()
X = tfidf.fit_transform(xcnt).toarray()
# print(X.shape)

all_words = vectorizer.get_feature_names()
n_words = len(all_words)

print('ALL WORDS : ', all_words)
# print(n_words)

X_df = pd.DataFrame.from_dict({w: X[:, i] for i, w in enumerate(all_words)})
print(X_df)


def sub_classifier(keyword):
    print('\nFor', keyword, 'data.')
    y = data[keyword].values

    # splitting the data into training and testing set
    xtrain, xtest, ytrain, ytest = train_test_split(X_df, y, test_size=0.2, random_state=10, stratify=y)
    # print(xtrain, "\n", xtest, '\n', ytrain, '\n', ytest)

    # Implementation with SGDClassifier Algorithm
    # SGDClassifier is stochastic gradient descent (SGD). This implementation works with data represented as dense or sparse arrays of floating point values for the features.
    print('\n Implementation with SGDClassifier Algorithm.')

    sgd = SGDClassifier(max_iter=5, tol=None)

    # fitting training data into the model
    sgd.fit(xtrain, ytrain)

    # predicting the test data
    ypred = sgd.predict(xtest)

    # accuracy of the model
    acc_sgd = round(sgd.score(xtrain, ytrain) * 100, 2)
    print('Training set Accuracy : ', acc_sgd, "%")
    print("Testing set Accuracy : ", round(accuracy_score(ytest, ypred) * 100, 2), '%')

    # Implementation with LogisticRegression Algorithm.
    print('\nImplementation with LogisticRegression Algorithm.')

    lr = LogisticRegression(max_iter=500)

    # fitting training data into the model
    lr.fit(xtrain, ytrain)

    # predicting the test data
    ypred = lr.predict(xtest)
    yprob = lr.predict_proba(xtest)

    # accuracy of the model
    print('Training set Accuracy : ', round(lr.score(xtrain, ytrain) * 100, 2), "%")
    print('Testing set Accuracy : ', round(accuracy_score(ytest, ypred) * 100, 2), '%')

    # Implementation with RandomForestClassifier Algorithm.
    print('\nImplementation with RandomForestClassifier Algorithm.')

    rfc = RandomForestClassifier(criterion='entropy', n_estimators=100)

    # fitting training data into the model
    rfc.fit(xtrain, ytrain)

    # predicting the test data
    ypred = rfc.predict(xtest)

    # accuracy of the model
    print('Training set Accuracy : ', round(rfc.score(xtrain, ytrain) * 100, 2), "%")
    print("Testing set Accuracy : ", round(accuracy_score(ytest, ypred) * 100, 2), '%')

    # Implementation with KNeighbourClassifier
    print('\nImplementation with KNeighbourClassifier')

    knn = KNeighborsClassifier(n_neighbors=10)

    # fitting training data into the model
    knn.fit(xtrain, ytrain)

    # predicting the test data
    ypred = knn.predict(xtest)

    # accuracy of the model
    print('Training set Accuracy : ', round(knn.score(xtrain, ytrain) * 100, 2), "%")
    print("Testing set Accuracy : ", round(accuracy_score(ytest, ypred) * 100, 2), '%')

    # Implementation with XGBClassifier
    print('\nImplementation with XGBClassifier')

    xgb = XGBClassifier()

    # fitting training data into the model
    xgb.fit(xtrain, ytrain, early_stopping_rounds=10,
            eval_metric="logloss",
            eval_set=[(xtest, ytest)], verbose=False)

    # predicting the test data
    ypred = xgb.predict(xtest)

    # accuracy of the model
    print('Training set Accuracy : ', round(knn.score(xtrain, ytrain) * 100, 2), "%")
    print("Testing set Accuracy : ", round(accuracy_score(ytest, ypred) * 100, 2), '%')

    # Implementation with keras.
    print("\n Implementation with keras.")

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

    # Model constants.
    max_features = 20000
    embedding_dim = 128
    sequence_length = 500

    # build the model
    print('Building the model.')

    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="float32")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    epochs = 20

    # Fit the model using the train and test datasets.
    model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=epochs)

    return xgb


fav_classifier = sub_classifier('fav_world')
info_classifier = sub_classifier('info')
decision_classifier = sub_classifier('decision')
structure_classifier = sub_classifier('structure')

# Plotting the accuracies
# As, per our post data plotting out what type of personalities do people have
fig3 = plt.figure(figsize=(10, 6))
# rcParams['figure.figsize'] = 10, 6

# creating a timer object and setting an interval of 3000 milliseconds
timer = fig3.canvas.new_timer(interval=3000)

timer.add_callback(close_event)

plt.subplots_adjust(wspace=0.5)

ax1 = plt.subplot(1, 4, 1)
plt.pie([sum(data['fav_world']),
         len(data['fav_world']) - sum(data['fav_world'])],
        labels=['Extrovert', 'Introvert'],
        explode=(0, 0.1),
        autopct='%1.1f%%')

ax2 = plt.subplot(1, 4, 2)
plt.pie([sum(data['info']),
         len(data['info']) - sum(data['info'])],
        labels=['Sensing', 'Intuition'],
        explode=(0, 0.1),
        autopct='%1.1f%%')

ax3 = plt.subplot(1, 4, 3)
plt.pie([sum(data['decision']),
         len(data['decision']) - sum(data['decision'])],
        labels=['Thinking', 'Feeling'],
        explode=(0, 0.1),
        autopct='%1.1f%%')

ax4 = plt.subplot(1, 4, 4)
plt.pie([sum(data['structure']),
         len(data['structure']) - sum(data['structure'])],
        labels=['Judging', 'Perceiving'],
        explode=(0, 0.1),
        autopct='%1.1f%%')

timer.start()
plt.show()
timer.stop()

# Feature Importance
# For fav_classifier that is
plot_importance(fav_classifier, max_num_features=20)
plt.title("Features associated with Extrovert")
plt.show()


# For fav_classifier that is
plot_importance(info_classifier, max_num_features=20)
plt.title("Features associated with Sensing")
plt.show()


# For fav_classifier that is
plot_importance(decision_classifier, max_num_features=20)
plt.title("Features associated with Thinking")
plt.show()


# For fav_classifier that is
plot_importance(structure_classifier, max_num_features=20)
plt.title("Features associated with Judging")
plt.show()
