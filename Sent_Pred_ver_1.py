from __future__ import division
import pandas as pd
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import Tkinter as tki
from Tkinter import *
import time

# Dataset
train_data_file_name = ".\\dataset\\Samsung_train.csv"
test_data_file_name = ".\\dataset\\Samsung_test.csv"
test_data_df = pd.read_csv(test_data_file_name, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_data_file_name, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]

# Preprocessing
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(analyzer='word', tokenizer=tokenize, lowercase=True, stop_words='english', max_features=100)

# vectorization (Term-Document Matrix)
corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist() + test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()

# Choose a classifier to train data. Comment classifiers that you are not applying
# Compute speed
start_time1 = time.time()
#Pre_model = LogisticRegression()
#Pre_model = SVC()
#Pre_model = MLPClassifier()
#Pre_model=GaussianProcessClassifier()
#Pre_model=KNeighborsClassifier()
#Pre_model=DecisionTreeClassifier()
#Pre_model=RandomForestClassifier()
#Pre_model=GaussianNB()
Pre_model=AdaBoostClassifier()

Pre_model = Pre_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
print "Classifier modeling - Lapsed time = ", time.time()-start_time1

root = tki.Tk()
root.geometry("265x300")
#root.configure(background="#FFA07A")

var = StringVar()
lab = Label(root, textvariable=var, font=("Georgia", 9))
var.set("Sentiment Prediction")
#lab.configure(bg="#FFA07A")
lab.pack(pady=5)

frm1 = tki.Frame(root, width=100, height=100)
#frm1.configure(background="green")
frm1.pack()

var = tki.StringVar(root)
var.set('Select feature')
# Few menu items (product features) are listed here, add more features as you like
choices = ["camera", "display", "battery", "price", "memory", "RAM", "speed", "connectivity", "FM"]
option = tki.OptionMenu(frm1, var, *choices)
option.configure(borderwidth=1, font=("Georgia", 9), height=1, width=10)
#option.pack()
option.grid(row=0, column=0, padx=2, pady=2)

# create a Frame for the Text and Scrollbar
txt_frm = tki.Frame(root, width=100, height=100)
#txt_frm.configure(background="green")
txt_frm.pack()

var1 = StringVar()
lable1 = Label(txt_frm, textvariable=var1, font=("Georgia", 9))
#lable1.configure(fg="green")
var1.set("Positive")

var2 = StringVar()
lable2 = Label(txt_frm, textvariable=var2, font=("Georgia", 9))
#lable2.configure(background="green")
var2.set("Negative")

var3 = StringVar()
lable3 = Label(txt_frm, textvariable=var3, font=("Georgia", 9))
#lable3.configure(background="green")
var3.set("Neutral")

lable1.grid(row=0)
lable2.grid(row=2)
lable3.grid(row=4)

# this step creates a Text widget
txt = tki.Text(txt_frm, bd=3, height=3, width=20)
txt.config(borderwidth=1, font=("Georgia", 9), undo=True, wrap='word')

# this step creates a Scrollbar and associate it with text box widget
scrollb = tki.Scrollbar(txt_frm, command=txt.yview)
scrollb.grid(row=1, column=2, sticky='nsew')
txt["yscrollcommand"] = scrollb.set

txt1 = tki.Text(txt_frm, bd=3, height=3, width=20)
txt1.config(borderwidth=1, font=("Georgia", 9), undo=True, wrap='word')
scrollb1 = tki.Scrollbar(txt_frm, command=txt1.yview)
scrollb1.grid(row=3, column=2, sticky='nsew')
txt1["yscrollcommand"] = scrollb1.set

txt2 = tki.Text(txt_frm, bd=3, height=3, width=20)
txt2.config(borderwidth=1, font=("Georgia", 9), undo=True, wrap='word')
scrollb2 = tki.Scrollbar(txt_frm, command=txt2.yview)
scrollb2.grid(row=5, column=2, sticky='nsew',)
txt2["yscrollcommand"] = scrollb2.set

def Predict_Sentiment():
    sf = "value is %s" % var.get()
    root.title(sf)

    # stemming the selected feature
    feat = stemmer.stem(var.get().lower())

    # getting test data from corpus_data_features_nd
    test = []
    ad = len(train_data_df)
    for x in range(0, len(test_data_df)):
        test.append(corpus_data_features_nd[x+ad])

    ind1 = []
    for i in range(0,len(test)):
        text = re.sub("[^a-zA-Z 0-9]", " ", test_data_df['Text'][i])
        # tokenizing the text
        tokens2 = nltk.word_tokenize(text)
        # Find whether the text containing the selected feature. Get index, if text containing selected feature
        if feat in tokens2:
            ind1.append(i)

    # Sentiment prediction from test data
    pos = 0
    pos_s = []
    neg = 0
    neg_s = []
    neu = 0
    neu_s = []
    for i in range(0, len(ind1)):
        rs = test[ind1[i]]
        sent_f = Pre_model.predict(np.array(rs).reshape(1,-1))
        if sent_f == '1':
            pos += 1
            pos_s.append(test_data_df["Text"][ind1[i]])
        elif sent_f == '0':
            neu += 1
            neu_s.append(test_data_df["Text"][ind1[i]])
        else:
            neg += 1
            neg_s.append(test_data_df["Text"][ind1[i]])
    # print results
    #print "Positive = ", pos
    #print "Positive reviews = ", pos_s
    #print "Neutral = ", neu
    #print "Neutral reviews = ", neu_s
    #print "Negative = ", neg
    #print "Negative reviews = ", neg_s

    # display results into appropriate text box
    txt.delete("1.0", END)
    txt.insert(INSERT, str(pos)+"; ")
    for x in pos_s:
        txt.insert(INSERT, x)
    txt.grid(row=1, column=1, sticky="nsew")

    txt1.delete("1.0", END)
    txt1.insert(INSERT, str(neg)+"; ")
    for x in neg_s:
        txt1.insert(INSERT, x)
    txt1.grid(row=3, column=1, sticky="nsew")

    txt2.delete("1.0", END)
    txt2.insert(INSERT, str(neu)+"; ")
    for x in neu_s:
        txt2.insert(INSERT, x)
    txt2.grid(row=5, column=1, sticky="nsew", padx=0, pady=0)

    # compute rating in range 0 to 1
    #if len(ind1)>0:
    #    print "Rating = ", (pos/len(ind1))*10
    #else:
    #    print "Rating = ", "0"

# create a submit button widget
b1 = tki.Button(frm1, text="Predict", command=Predict_Sentiment, font=("Georgia", 9))
b1.configure(borderwidth=1, height=1, width=7)
b1.grid(row=0, column=1, padx=3, pady=1)

txt.grid(row=1, column=1, sticky="nsew", padx=1, pady=1)
txt1.grid(row=3, column=1, sticky="nsew", padx=1, pady=1)
txt2.grid(row=5, column=1, sticky="nsew", padx=1, pady=1)

tki.mainloop()
