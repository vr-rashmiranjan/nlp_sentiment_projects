import pandas as pd
import numpy as np

import seaborn as sns

import neattext.functions as nfx

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

import joblib

raw_data=pd.read_csv('data\sentiment_data.csv')

data=raw_data.copy()
# print(data.head())
#
# data.shape
#
# data.columns

data['Emotion'].value_counts()

sns.countplot(x='Emotion',data=data)

dir(nfx)

data['clr_text']=data['Text'].apply(nfx.remove_userhandles)
data['clr_text']=data['clr_text'].apply(nfx.remove_stopwords)

data.head()

x=data['clr_text']
y=data['Emotion']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42 )

pipe_lr=Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])

pipe_lr.fit(x_train,y_train)


pipe_lr.score(x_test,y_test)

'''
st='i am not good'

pipe_lr.predict([st])

pipe_lr.predict_proba([st])
'''
## saving the model

model_file=open("emo_class_model_27aug.pkl","wb")
joblib.dump(pipe_lr,model_file)
model_file.close()


