from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk import RegexpTokenizer
import gensim.downloader as api
import pandas as pd
import numpy as np
import logging
import pickle
import os
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""
IMPORTANT NOTE:

This script creates a fitted, pickled LogisticRegression classifier for the Flask app to use.

This file is not required to be run for flask app operation, as long as there exists pickle
file "lr_model_weighted_tfidf.model" in the same dir as app.py.

You can run this file to re-create the pickle file if required. However the submission
will contain the finished pickle file and should be good to go.

This makes use of the gensim pretrained W2V model so will take some time to complete.

"""


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
np.random.seed(0)

CATEGORIES = ['Engineering', 'Accounting_Finance', 'Healthcare_Nursing', 'Sales']

tokeniser = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

# Load stop words
with open("static/data/stopwords_en.txt") as file:
    stopwords = file.read().splitlines()


# Load and parse job ad text files then return a dictionary of ads where k=job id, v=ad data.
def load_saved_ads_from_file(categories: list, stopwords):
    ads = {}
    for cat in CATEGORIES:

        # Load text from file:
        path = "static/data/" + cat + "/"
        for filename in os.listdir(path):
            file = open(path + filename, encoding="utf-8")
            contents = file.read().split("\n")
            job_id = int(re.findall('\d+', filename)[0])  # noqa
            ad = {"job_id": job_id, "category": cat}
            for line in contents:
                tokens = line.split(":")
                if tokens[0].upper() == "TITLE":
                    ad['title_original'] = tokens[1:][0]
                elif tokens[0].upper() == "WEBINDEX":
                    try:
                        ad['webindex'] = int(tokens[1])
                    except ValueError:
                        ad['webindex'] = None
                elif tokens[0].upper() == "COMPANY":
                    ad['company'] = tokens[1]
                elif tokens[0].upper() == "DESCRIPTION":
                    ad['desc_original'] = "".join(tokens[1:])

            # Close file and store finished ad data
            file.close()
            ads[job_id] = ad

    return ads


# Load ads from file
ads = load_saved_ads_from_file(CATEGORIES, stopwords)
# print(json.dumps(ads, indent=2))

# Identify 50 most frequent and single occurrence terms with respect to entire corpus
# where corpus = all job ads description + titles
n = 50
freq = {}
title_desc_joined = [str(v['title_original'] + " " + v['desc_original']) for v in ads.values()]
documents_joined = [re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", doc.lower()) for doc in title_desc_joined]
for index, document in enumerate(documents_joined):
    for term in documents_joined[index]:
        try:
            freq[term] += 1
        except KeyError:
            freq[term] = 1
single_occ_terms = [k for k, v in freq.items() if v == 1 or v == 2]
common_terms_desc = [k for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)]

# Remove single occurrence and n-common terms from each document
n_common = [t for t in common_terms_desc[:n]]
for index, document in enumerate(documents_joined):
    new_doc = [t for t in documents_joined[index] if t not in single_occ_terms or t not in n_common]
    documents_joined[index] = new_doc

# Re-join terms and create vocab ready for next step, save vocab to file
documents_joined = [" ".join(t) for t in documents_joined]
vocab = sorted(list(freq.keys()))
with open('static/data/vocab.txt', 'w') as file:
    for line in vocab:
        file.write(f"{line}\n")

# Create TF-IDF weightings
tVectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocab, ngram_range=(1, 2))
tfidf_features = tVectorizer.fit_transform(documents_joined)

# Store weights as {word:weight} maps for next stage.
tfidf_weights = []
for index, record in enumerate(documents_joined):
    vector = {}
    for word, value in zip(vocab, tfidf_features.toarray()[index]):
        if value > 0:
            vector[word] = str(value)
    tfidf_weights.append(vector)

# Load pre-trained w2v model
preT_W2v = api.load('word2vec-google-news-300')

# Using pretrained W2V, create weighted embedding feature representation.
weighted_vectors = pd.DataFrame()
for index in range(0, len(documents_joined)):

    # Get the processed description for the current ad
    tokens = documents_joined[index].split()
    temp = pd.DataFrame()

    # Iterate individual words per document
    for word_index in range(0, len(tokens)):
        try:
            # Get the vector for each word using the pretained W2V model
            word = tokens[word_index]
            word_vec = preT_W2v[word]
            word_weight = float(tfidf_weights[index][word])

            # Save word to temp DF if word is present
            temp = temp.append(pd.Series(word_vec * word_weight), ignore_index=True)
        except:
            pass

    weighted_vectors = weighted_vectors.append(temp.sum(), ignore_index=True)

# Fit LR classifier with entire dataset (as we will be making predictions on new data, not testing models here)
labels = [v['category'] for v in ads.values()]
lr_model_weighted_tfidf = LogisticRegression(random_state=30, max_iter=1500)
lr_model_weighted_tfidf.fit(weighted_vectors.to_numpy(), labels)

# Save model to file
filename = 'lr_model_weighted_tfidf.model'
pickle.dump(lr_model_weighted_tfidf, open(filename, 'wb'))


# Return predicted category based on the input arguments
def predict_category(title, description, model):

    # Join title and description, tokenise and lowercase
    text = [title + " " + description]
    text = [re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", t.lower()) for t in text][0]

    # Remove single occurrence and n-common terms
    text = [t for t in text if t not in single_occ_terms or t not in n_common]
    text = [' '.join(text)]

    # Create TF-IDF weightings
    tVectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocab, ngram_range=(1, 2))
    tfidf_features = tVectorizer.fit_transform(text)

    # Store weights as {word:weight} maps for next stage.
    tfidf_weights = []
    for index, record in enumerate(text):
        vector = {}
        for word, value in zip(vocab, tfidf_features.toarray()[index]):
            if value > 0:
                vector[word] = str(value)
        tfidf_weights.append(vector)

    # Using pretrained W2V, create weighted embedding feature representation.
    weighted_vectors = pd.DataFrame()
    for index in range(0, len(text)):
        tokens = text[0].split()
        temp = pd.DataFrame()
        for word_index in range(0, len(tokens)):
            try:
                # Get the vector for each word using the pretained W2V model
                word = tokens[word_index]
                word_vec = preT_W2v[word]
                word_weight = float(tfidf_weights[index][word])

                # Save word to temp DF if word is present
                temp = temp.append(pd.Series(word_vec * word_weight), ignore_index=True)
            except:
                pass

        weighted_vectors = weighted_vectors.append(temp.sum(), ignore_index=True)

    return model.predict(weighted_vectors.to_numpy())


# Test the model on some ads
# print(predict_category(ads[631]["title_original"], ads[631]["desc_original"], lr_model_weighted_tfidf))
# print(predict_category(ads[632]["title_original"], ads[632]["desc_original"], lr_model_weighted_tfidf))
# print(predict_category(ads[633]["title_original"], ads[633]["desc_original"], lr_model_weighted_tfidf))
# print(predict_category(ads[634]["title_original"], ads[634]["desc_original"], lr_model_weighted_tfidf))
# print(predict_category(ads[635]["title_original"], ads[635]["desc_original"], lr_model_weighted_tfidf))
