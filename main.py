import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


stopwords = stopwords.words('english')
# IF NEEDING TO REMOVE WORDS ADD THOSE WORDS TO THE STOPWORDS LIST
new_words = ["ya"]
stopwords.append(new_words)

train_data = open("trainhw1new.txt", "r")
test_data = open("testdatahw1.txt", "r")

# MAYBE INSTEAD OF POSITIVE AND NEGATIVE USE XTRAIN YTRAIN
# X_TRAIN BEING THE REVIEWS, Y_TRAIN BEING 1 pos 0 neg


fdX_train = []

fdY_train = []

X_test = []

"""
First clean the data; remove punctuation etc
EACH LINE IS A REVIEW IN ITSELF
Procedure is to clean the data, remove the punctuation, set to lowercase, and filter out stopwords to remove noise
Then sort the training data based on whether it is a positive or negative review in order
To preform the Term Frequency Calculations
"""
#i = 0
for line in train_data:
    # Based on pos or neg add 1, 0 to Y_train the ï is for the first review
    if line[0] == "1" or line[0] == "ï":
        fdY_train.append(1)
    else:
        fdY_train.append(0)
    # For each line remove special characters and numbers remove stop words and create formatted data
    line = line.translate(str.maketrans('', '', r"!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~ï»¿-0123456789"))
    word_tokens = word_tokenize(line.lower())
    new_line = []
    for word in word_tokens:
        if word not in stopwords:
            new_line.append(word)

    new_line = ' '.join(new_line)
    # Add to X_train
    fdX_train.append(new_line)
#    i += 1
#    if i == 50:
#        break

# Repeat above for test data only adding to X_test
#j = 0
for line in test_data:
    line = line.translate(str.maketrans('', '', r"!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~ï»¿-0123456789"))
    word_tokens = word_tokenize(line.lower())
    new_line = []
    for word in word_tokens:
        if word not in stopwords:
            new_line.append(word)
    new_line = ' '.join(new_line)
    X_test.append(new_line)
#    j += 1
#    if j == 50:
#        break

# Create the data frame of X and Y tests and trains
vectorizer = TfidfVectorizer(min_df=10, max_df=0.65)

# applying tf idf to training data
X_train_tf = vectorizer.fit_transform(fdX_train)
X_train_tf = vectorizer.transform(fdX_train)

# The TFIDF of the test and train data
X_test_tf = vectorizer.transform(X_test)

X_train_tf_Array = X_train_tf.toarray()
X_test_tf_Array = X_test_tf.toarray()

print(X_train_tf.shape)


cx = lambda a, b : np.inner(a, b)/(LA.norm(a)*LA.norm(b))
i = 0
for vector in X_train_tf_Array:
    for testV in X_test_tf_Array:
        cosine = cx(vector, testV)
        if cosine == 0:
            i += 1
            print(i)



