import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import nltk
import string

df=pd.read_csv("news/NEWS.csv")
df['combined']=df['Statement']+''+df['Link']+''+df['Label']
print(df.head())
#DataFlair - Get the labels
labels=df.Label
labels.head()
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['Statement'],labels, test_size=0.2, random_state=7)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

for (label), group in df.groupby(['Label']):
     group.to_csv(f'result/{label}.csv', index=False)
 