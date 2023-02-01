
#Natural Language Toolkit: Utility Functions
#provides various text processing libraries with lot of dataset,tasks perform such as tokenizing.
#Tokenization is the process of replacing sensitive data with unique identification symbols that 
# retain all the essential information about the data without compromising its security.
from nltk.util import pr

# pandas is a Python package providing fast, flexible, and expressive data structures 
# It is designed to make working with “relational” or “labeled” data both easy and intuitive.
import pandas as pd

# numpy provides a multidimensional array object, as well as variations such as masks and matrices,
#  which can be used for various math operations.
import numpy as np

#tool provided by the scikit-learn library
#used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
from sklearn.feature_extraction.text import CountVectorizer

#spilts array or matrices into random subsets for train and test data
from sklearn.model_selection import train_test_split

#Decision Tree-supervised learning method used for classification.
#goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from data features.
from sklearn.tree import DecisionTreeClassifier

#extract a dataset
data = pd.read_csv("twitter.csv")
#print(data.head())

#add a new column to the dataset as labels which will contain the value as :
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

#Only selct the tweet and labels columns for rest of the task of training a model
data = data[["tweet", "labels"]]
#print(data.head())

#provides a set of powerful regular expression facilities,
# which allows you to check whether a given string matches a given pattern, or contains such a pattern
import re

#nltk toolkit was created for users to deal with NLP.
# it offers numerous test data as data set and different text prcessing librarires.
import nltk

#This module provides a port of the Snowball 
#Stemming is the process of reducing inflection in words to their root forms 
# such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.
stemmer = nltk.SnowballStemmer("english")

# They are words that you do not want to use to describe the topic of your content. 
# They are pre-defined and cannot be removed.
from nltk.corpus import stopwords
#It's a built-in module and we have to import it before using any of its constants and classes
import string
stopword=set(stopwords.words('english'))

#Create a function to clean the texts in the tweet column
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

#split the dataset into training and test sets and train a machine learning model for the task
x = np.array(data["tweet"])
y = np.array(data["labels"])

#transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text
cv = CountVectorizer()

# used on the training data so that we can scale the training data and also learn the scaling parameters of that data
# it will calculate the mean(μ) and standard deviation(σ) of the feature F at a time it will transform the data points of the feature F
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#A classifier algorithm is used to map input data to a target variable through decision rules.
#  Can be used to predict and understand what characteristics are associated with a specific class or target.
clf = DecisionTreeClassifier()

#evaluate how the classifier performs on the training and test set with . score
clf.fit(X_train,y_train)

#calculates the F1 score for a set of predicted labels. 
# An F1 score can range between 0 − 1 0-1 0−1, with 0 being the worst score and 1 being the best.
clf.score(X_test,y_test)

#create a function to find whether the speech is hate speech or not
def hate_speech_detection():
#Streamlit is an open source app framework in Python language.
#  It helps us create web apps for data science and machine learning in a short time
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
hate_speech_detection()