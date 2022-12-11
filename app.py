from flask import Flask, render_template, redirect, request
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import RegexpTokenizer
import gensim.downloader as api
import pandas as pd
import logging
import pickle
import os
import re


# Set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load pretrained W2V model - this will take some time and site be unresponsive until loaded.
print("Loading W2V model - this may take a while. Site will be unresponsive until this process completes...")
w2v = api.load('word2vec-google-news-300')

# Load pickled LR classifier and vocab
logging.info("Loading LR classifier from file..")
lr_model_weighted_tfidf = pickle.load(open("lr_model_weighted_tfidf.model", 'rb'))
with open("static/data/vocab.txt") as file:
    vocab = file.read().splitlines()


# Loads and parses job ad text files then returns a dictionary of ads where k=job id, v=ad data.
def load_saved_ads_from_file(categories: list):
    ads = {}
    for cat in categories:
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


# Return predicted category based on the input arguments
def predict_category(title, description, model, vocab):

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
                # Get vector for each word using the pretained W2V model
                word = tokens[word_index]
                word_vec = w2v[word]
                word_weight = float(tfidf_weights[index][word])

                # Save word if word is present
                temp = temp.append(pd.Series(word_vec * word_weight), ignore_index=True)
            except:  # noqa
                pass

        weighted_vectors = weighted_vectors.append(temp.sum(), ignore_index=True)

    return model.predict(weighted_vectors.to_numpy())


# Load ads from file
logging.info("Loading stored ads..")
CATEGORIES = ['Engineering', 'Accounting_Finance', 'Healthcare_Nursing', 'Sales']
ads = load_saved_ads_from_file(CATEGORIES)

# Load stopwords from file
logging.info("Loading stopwords..")
with open("static/data/stopwords_en.txt") as file:
    stopwords = file.read().splitlines()

# Initialise tokeniser object
tokeniser = RegexpTokenizer(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

# Identify n most frequent and single occurrence terms with respect to entire corpus
# These will be removed from incoming title + description texts being classified
logging.info("Identifying frequency-based term removals..")
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
n_common = [t for t in common_terms_desc[:n]]


# Start Flask app
logging.info("Starting Flask app..")
app = Flask(__name__)
app.debug = False
app.use_reloader = False
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
app.config['TESTING'] = False


# Homepage/index route
@app.route('/')
def index():
    return redirect('/All')


# Category view endpoint. Displays all ads under a certain category
@app.route('/<cat>')
def browse(cat):
    return render_template('browse.html', ads=ads, cat=cat, categories=CATEGORIES)


# Endpoint showing newly created ads. Use this to easily check created ads
@app.route('/recent')
def recent():

    # Ads without a webindex will serve as "recently listed"
    recent_ads = {k: v for k, v in ads.items() if v['webindex'] is None}
    total_recent_ads = len(recent_ads.items())
    return render_template('recent.html', ads=recent_ads, total=total_recent_ads)


# Detailed view endpoint. Displays full description of a given ad
@app.route('/ad/<job_id>')
def view_ad(job_id):
    return render_template('view_ad.html', ads=ads, job_id=int(job_id), message=None)


# Ad creation endpoint. Initiates the process of posting new ads.
@app.route('/create/', methods=['GET', "POST"])
def create():
    total_ads = len(ads.items())

    # GET request returns the job input form
    if request.method == 'GET':
        return render_template('create.html', ads=ads, total=total_ads, error=None)

    # POST request returns confirmation screen, or error message if inputs were invalid
    if request.method == 'POST':

        title = request.form.get('title')
        company = request.form.get('company')
        description = request.form.get('description')

        # Load confirmation page if all required fields had input
        if len(title) > 0 and len(description) > 0:

            # predict catefory from title and description texts
            predicted_category = predict_category(title, description, lr_model_weighted_tfidf, vocab)[0]

            # User will be prompted to confirm predicted category or choose their own
            return render_template('confirm_create.html', ads=ads, total=total_ads, title=title, company=company, description=description, predicted_category=predicted_category, categories=CATEGORIES)

        # Error message if title or description were null
        else:
            return render_template('create.html', ads=ads, total=total_ads, error="Error: Title and Description fields must not be blank")


@app.route('/confirm/', methods=["POST"])
def confirm():
    if request.method == 'POST':

        # Get final ad data from form submission
        title = request.form.get('title')
        company = request.form.get('company')
        description = request.form.get('description')
        category = request.form.get('select_category')

        # Save new ad to ads json
        new_job_id = len(ads.keys()) + 1
        ads[new_job_id] = {
            "job_id": new_job_id,
            "category": category,
            'title_original': title,
            'webindex': None,
            'company': company,
            'desc_original': description
        }

        # Save new ad to file
        path = "static/data/" + category + "/"
        filename = "Job_" + str(new_job_id).zfill(5) + ".txt"
        body = [
            "Title: " + title + "\n",
            "Webindex: N/A" + "\n",
            "Company: " + company + "\n",
            "Description: " + description + "\n"
        ]

        with open(path + filename, 'w', encoding="utf-8") as file:
            for line in body:
                file.write(line)

        # Display a view of the new ad with a success message
        return render_template('view_ad.html', ads=ads, job_id=new_job_id, message="Congratulations! Your ad was successfully created. View it in the 'Recently Posted', or in category view.")

