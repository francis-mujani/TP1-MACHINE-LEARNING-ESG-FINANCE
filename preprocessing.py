import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Label Encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Model Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import string
import re
import nltk
import nltk.corpus
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

import joblib


print('CREATE DATAFRAME...')

def create_data():
    positif = pd.read_pickle(r'data/imdb_raw_pos.pickle')
    negatif = pd.read_pickle(r'data/imdb_raw_neg.pickle')
    pos = {'Critiques': positif}
    neg = {'Critiques': negatif}

    # Creates pandas DataFrame.
    df1 = pd.DataFrame(pos)
    df1['Sentiment'] = 'positif'
    df1['Reponse'] = 1
    df2 = pd.DataFrame(neg)
    df2['Sentiment'] = 'Negatif'
    df2['Reponse'] = 0
    print('CREATE DATAFRAME DONE!')

    ### CONCAT DATA
    data = pd.concat([df1, df2])
    # SHUFFLED DATA
    shuffled = data.sample(frac=1).reset_index()
    shuffled = shuffled.drop(['index'], axis=1)
    print('CONCAT DATAFRAME DONE!')

    return shuffled

data = create_data()

print("CLEANING DATA...")
def Text_Cleaning(Text):
  # Lowercase the texts
  Text = Text.lower()

  # Cleaning punctuations in the text
  punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  Text = Text.translate(punc)

  # Removing numbers in the text
  Text = re.sub(r'\d+', '', Text)

  # Remove possible links
  Text = re.sub('https?://\S+|www\.\S+', '', Text)

  # Deleting newlines
  Text = re.sub('\n', '', Text)

  return Text
print("CLEANING DATA DONE")

print("TEXTING PREPOCESSING")
Stopwords = set(nltk.corpus.stopwords.words("english")) - set(["not"])
def Text_Processing(Text):
  Processed_Text = list()
  Lemmatizer = WordNetLemmatizer()

  # Tokens of Words
  Tokens = nltk.word_tokenize(Text)

  for word in Tokens:
    if word not in Stopwords:
      Processed_Text.append(Lemmatizer.lemmatize(word))

  return(" ".join(Processed_Text))


print('PREPROCESING...')
#### PREPROCESING
X = shuffled.Critiques
y = shuffled.Reponse

# # SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)


#print('longuer de X train :', len(X_train))
#print('longuer de y train :', len(y_train))
#print('longuer de X test :', len(X_test))
#print('longuer de y test :', len(y_test))
print('PREPROCESING DONE')


print('VECTORISATION...')
## VECTORISATION
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
vect.fit(X_train)
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)
print('VECTORISATION DONE')


print('TRAINING MODEL...')
# Create model MultinomialNB
NB = MultinomialNB()
NB.fit(X_train_vect, y_train)
print('TRAINING MODEL DONE')

#save model
joblib.dump(NB, 'models/NB.model')
