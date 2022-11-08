from flask import Flask
import os
from flask import url_for
from flask import render_template
import joblib
import pandas as pd
from flask import Flask, url_for, request, render_template, redirect

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import vect

app = Flask(__name__)
app.config["DEBUG"] = True
##



#load saved model
model = joblib.load('models/LG.model')
data = pd.read_csv('model_result.csv')

@app.route("/")
def root():
    module_path = os.path.dirname(os.path.dirname(os.path.dirname('TP1')))
    print('jfjfj', module_path)
    print('render template')
    return render_template('index.html', data=data)

@app.route("/", methods=["POST"])
def comment():

    comment = request.form['comment']
    print('COMMENTAIRE :', comment)
    test = []
    test.append(comment)
    test_vect = vect.transform(test)
    print(test_vect)
    predLabel = model.predict(test_vect)

    tags = ['Negative', 'Positive']
    predict = ''
    comment_F = ''
    comment_T = ''
    if len(comment) == 0 or (len(comment) == 2 or comment == " "):
        print('Comment', comment)
        predict = 'Rentrez une phrase ou un mot en anglais! SVP'
        comment_F = False
    else:
        predict = tags[predLabel[0]]
        comment_T = True
    print("Le model à prédit que votre commentaire est ", tags[predLabel[0]])
    return render_template('index.html', predict=predict, data=data, comment_F =comment_F, comment_T = comment_T)


# create port
port = int(os.environ.get("PORT", 5001))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=port)