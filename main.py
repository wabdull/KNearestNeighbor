import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

"""
First clean the data; remove punctuation etc
EACH LINE IS A REVIEW IN ITSELF
Procedure is to clean the data, remove the punctuation, set to lowercase, and filter out stopwords to remove noise
Then sort the training data based on whether it is a positive or negative review in order
To preform the Term Frequency Calculations
"""


def format_train(file, stopwords):
    x = []
    y = []
    for line in file:
        # Based on pos or neg add 1, -1 to Y_train the ï is for the first review
        if line[0] == "1" or line[0] == "ï":
            y.append("+1")
        else:
            y.append("-1")
        # For each line remove special characters and numbers remove stop words and create formatted data
        line = line.translate(str.maketrans('', '', r"!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~ï»¿-0123456789"))
        # Set the words to lowercase
        word_tokens = word_tokenize(line.lower())
        new_line = []
        for word in word_tokens:
            if word not in stopwords:
                # If the word isn't a stop word keep it
                new_line.append(word)

        new_line = ' '.join(new_line)
        # Add to X_train
        x.append(new_line)
    return x, y


# Repeat above for test data only adding to X_test
def format_test(file, stopwords):
    x_test = []
    for line in file:
        line = line.translate(str.maketrans('', '', r"!\"#$%&'()*+,./:;<=>?@[\]^_`{|}~ï»¿-0123456789"))
        word_tokens = word_tokenize(line.lower())
        new_line = []
        for word in word_tokens:
            if word not in stopwords:
                new_line.append(word)
        new_line = ' '.join(new_line)
        x_test.append(new_line)
    return x_test


def knn(x_train, y_train, x_test):
    output = []
    # Create the data frame of X and Y tests and trains
    vectorizer = TfidfVectorizer(min_df=25, max_df=0.85)

    # applying tf idf to training data
    x_train_tf = vectorizer.fit_transform(x_train)
    x_train_tf = vectorizer.transform(x_train)

    # The TFIDF of the test and train data
    x_test_tf = vectorizer.transform(x_test)

    x_train_tf_array = x_train_tf.toarray()
    x_test_tf_array = x_test_tf.toarray()

    for line in cosine_similarity(x_test_tf_array, x_train_tf_array):
        nearest = np.argsort(line)[-250:].tolist()
        labels = [y_train[i] for i in nearest]
        most_common = Counter(labels).most_common(1)
        output.append(most_common[0][0])
        print(most_common[0][0])
    return output


if __name__ == '__main__':
    stop_words = stopwords.words('english')
    # IF NEEDING TO REMOVE WORDS ADD THOSE WORDS TO THE STOPWORDS LIST
    new_words = ["ya"]
    stop_words.append(new_words)

    # Files of train and test data
    train_data = open("trainhw1new.txt", "r")
    test_data = open("testdatahw1.txt", "r")

    X_train, Y_train = format_train(train_data, stop_words)
    X_test = format_test(test_data, stop_words)
    KNNVALUES = knn(X_train, Y_train, X_test)

    out_file = open("output.txt", "w")
    for value in KNNVALUES:
        out_file.write(value)
        out_file.write("\n")
    out_file.close()

