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
from sklearn.preprocessing import MaxAbsScaler
from sklearn import metrics
import joblib




print('CREATE DATAFRAME...')
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
print('CREATE CONCAT DATAFRAME DONE!')



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

print('SCALER DATA')
transformer = MaxAbsScaler().fit(X_train_vect)
transformer_Xtrain = transformer.transform(X_train_vect)
print('SCALER DATA DONE')


print('TRAINING MODEL...')
Classifier = LogisticRegression(random_state = 0, C = 0.3906939937054613, penalty = 'l2')
Classifier.fit(transformer_Xtrain, y_train)
print('model saved')
joblib.dump(Classifier, 'models/LG.model')

# Create model MultinomialNB
#NB = MultinomialNB()
#NB.fit(X_train_vect, y_train)
print('TRAINING MODEL DONE')
y_pred = Classifier.predict(X_test_vect)
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

