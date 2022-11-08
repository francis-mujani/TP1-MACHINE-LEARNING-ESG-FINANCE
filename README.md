# TP 1 MACHINE LEARNING


### Comment lancer le projet
![alt app](./img.png)
    - taper sur le terminal la commande suivante:

```bash
 .\run.bat
```
## REALISATION DU PROJET TP1 ML
 - voir plus des details sur l'entrainement et l'evaluation des modeles dans le notebook
#### 1. Importation de données
```python
import pandas as pd

positif = pd.read_pickle(r'data/imdb_raw_pos.pickle')
negatif = pd.read_pickle(r'data/imdb_raw_neg.pickle')
```
### 2. Preprocesing
```python
import re
def Cleaner(mot):
  mot = mot.lower()
  punctuat = str.maketrans(string.punctuation, ' '*len(string.punctuation))
  mot = mot.translate(punctuat)
  mot = re.sub(r'\d+', '', mot)
  mot = re.sub('https?://\S+|www\.\S+', '', mot)
  mot = re.sub('\n', '', mot)

  return mot
```
### 3. Modelisation
```python
DTree = DecisionTreeClassifier()
LogReg = LogisticRegression()
SVC = SVC()
RForest = RandomForestClassifier()
Bayes = BernoulliNB()
KNN = KNeighborsClassifier()
```
### 4. Hyperparameter With GridSearch
```python
grid_search.fit(X_train_vect, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
```


