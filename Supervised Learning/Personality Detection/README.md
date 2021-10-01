## Introduction

- The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis : 
  - Introversion (I) – Extroversion (E)
  - Intuition (N) – Sensing (S)
  - Thinking (T) – Feeling (F)
  - Judging (J) – Perceiving (P)

- So for example, someone who prefers introversion, intuition, thinking and perceiving would be labelled an INTP in the MBTI system, and there are lots of personality based components that would model or describe this person’s preferences or behaviour based on the label.

- It is one of, if not then, the most popular personality test in the world. It is used in businesses, online, for fun, for research and lots more. A simple google search reveals all the different ways the test has been used over time. It’s safe to say that this test is still very relevant in the world in terms of its use.

## Database Content

This dataset contains over 8600 rows of data, on each row is a person’s:

- Type (This is person's 4 letter MBTI code/type)
- A section of each of the last 50 things they have posted (Each entry separated by "|||" (3 pipe characters)

## Goal

- **Goal** is to create new columns based on the content of the tweets, in order to create a predictive model.

## Implementation Steps

- Import all the required libraries.
- Loading the data using `pandas.read_csv(path)` function.
- Taking out info of the data using `data.info()` function. It will give the info of total non-null values in our data per column and type of data in the column.
- Describing out the data to know total number of values per column and all its unique value counts. Function is `data.describe()`.
- Type count of type column to know count of unique types.
- Plotted the above unique type count using matplotlib library.
- Set the timer for auto closing the graph.
- We have multiple posts with one post column. So, separating those posts into the lists and saved it in new column named 'separated_posts'.
- A new column named 'num_posts' contains the data with total number of posts within one post and plotted it into graph. 
- Separated posts are in list form so converting them to string so to pass that column in clean_data() for cleaning data.
- Creating the `clean_data()` function. This function will clean our data that is it will perform all following operations.
  - Lowercase the data and remove the links in the data.
  - Removing numbers.
  - Removing punctuations.
  - Removing extra white spaces.
- Now, tokenizing the data. This will change text or sting type data to list of words. For that, used the nltk's `word_tokenize()` function.
- Next step is to remove all the stop words from our data. Stop words are the most common words sin text like 'the, is, if, of, and',etc.
- Normalizing the data using `WordNetLemmatizer()` function of nltk library.
- Now, again converting the cleaned data into string data and saving it into `data['cleaned_data']`.
- Now, manipulating the type data.
- As, we have 8 main expressions divided into 4 with their opposite side as mentioned in beginning. Introversion, extroversion, intuition, sensing, thinking, feeling, judging, perceiving.
- For fav_world, if 'E' then 1 else 0 i.e for 'I'.
- Similarly, for other 3 personality groups.
- Next step is vectorizing the data.
- Used, CountVectorizer and TfidfVectorizer for vectorizing the data.
- Next is splitting the data into x and y and then training set and testing set.
- Making a new df i.e X_df, that contains all the unique words as its column title and its occurrence per post data that we get from tfidf vectorization.
- Now fitting our vectorized data into different model
- Models used:
  - SGDClassifier
  - Logistic Regression
  - Random Forest Classifier
  - K Neighbour Classifier
  - XGBClassifier
  - Keras model