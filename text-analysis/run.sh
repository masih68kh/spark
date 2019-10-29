#! /usr/bin/bash
# rm -r results/*
# python TextAnalyzer.py TF masc_500k_texts/written/fiction/hotel-california.txt results/hotel-california.tf
# python TextAnalyzer.py TF masc_500k_texts/written/twitter/tweets1.txt results/tweets1.tf
# python TextAnalyzer.py TFIDF results/hotel-california.tf results/hotel-california.tfidf --idfvalues anc.idf
# python TextAnalyzer.py TFIDF results/tweets1.tf results/tweets1.tfidf --idfvalues anc.idf
# python TextAnalyzer.py SIM results/hotel-california.tfidf cosSim_hotel-california_tweets1 --other results/tweets1.tfidf
python TextAnalyzer.py TF masc_500k_texts/written/fiction results/fiction.tf
python TextAnalyzer.py TF masc_500k_texts/written/spam results/spam.tf
python TextAnalyzer.py TF masc_500k_texts/spoken/face-to-face results/face-to-face.tf

python TextAnalyzer.py TFIDF results/fiction.tf results/fiction.tfidf --idfvalues anc.idf
python TextAnalyzer.py TFIDF results/spam.tf results/spam.tfidf --idfvalues anc.idf
python TextAnalyzer.py TFIDF results/face-to-face.tf results/face-to-face.tfidf --idfvalues anc.idf

python TextAnalyzer.py SIM results/face-to-face.tfidf face-spam.txt --other results/spam.tfidf
python TextAnalyzer.py SIM results/spam.tfidf spam-face.txt --other results/face-to-face.tfidf

python TextAnalyzer.py SIM results/face-to-face.tfidf face-fiction.txt --other results/fiction.tfidf
python TextAnalyzer.py SIM results/fiction.tfidf fiction-face.txt --other results/face-to-face.tfidf

python TextAnalyzer.py SIM results/spam.tfidf spam-fiction.txt --other results/fiction.tfidf
python TextAnalyzer.py SIM results/fiction.tfidf fiction-spam.txt --other results/spam.tfidf