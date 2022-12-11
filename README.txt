First unzip contents of static.zip into new directory named static.

Start app with "flask run --no-reload". 

Important to have no reload and no debug mode or app will hang for long periods as the W2V model reloads with each change.

The site will appear to be stuck or not loading during its initialisation stage as it is loading the W2V model which takes some time.

The file "create_fit_classifier.py" does not need to be run for the Flask app to function. You may re-run it to re-create "lr_model_weighted_tfidf.model" pickle file.

Developed and tested in a conda venv running Python version 3.9.5

