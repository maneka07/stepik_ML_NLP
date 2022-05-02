# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:17:32 2022

@author: Tatiana
"""

import io
import csv
import nltk
import re
import string
import numpy as np
from copy import deepcopy
texts = []
summs = []



with io.open('news.csv', encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        texts.append(row[0])
        summs.append(row[1])
        
#text = "I love, dog. It's not my dog? Dog loves me!!!"
print(len(texts))
textid = 1
text = texts[textid]
sentences = re.split("[.?!]", text.strip())
table = str.maketrans('','', string.punctuation)
stopw = set(nltk.corpus.stopwords.words('russian'))
sent_toks = []
stemmer = nltk.stem.porter.PorterStemmer()
for i, s in enumerate(sentences): 
    if len(s) == 0:
        continue
    toks = nltk.word_tokenize(s)
    sent_toks.append([stemmer.stem(w.lower()) for w in toks if len(w) > 2 and w not in stopw])

mat = np.zeros((len(sent_toks), len(sent_toks)))
for i, s1 in enumerate(sent_toks):
    for j, s2 in enumerate(sent_toks[i+1:]):
        cnt = len([w for w in s1 if w in s2])
        mat[i][i+1+j] = mat[i+1+j][i] = cnt

p_tele = 0.15

mat = [[x/sum(y) if sum(y) > 0 else 1/len(y) for x in y] for y in mat]
print(mat)
mat =[[ x*(1-p_tele) + p_tele/len(mat) for x in y] for y in mat]
mat = np.array(mat)
np.set_printoptions(precision=4)
state = np.zeros(len(sent_toks))
state[0] = 1.0
for i in range(200): 
    old_state = deepcopy(state)
    state = mat.transpose(1,0).dot(state)
    #print(state-old_state)
    #print(i, (state-old_state).sum())
    #print(old_state, state)
    if (state == old_state).all():
        print ("converged at ", i)
        break

textrank = sorted([(k,v) for k, v in enumerate(state)], key=lambda x: x[1], reverse=True)
 
print(textrank)
summ ="" 
for i in range(3):
    summ += sentences[textrank[i][0]] + "."
print(summ)
print("-----")
print(summs[textid])
#from rouge_score import rouge_scorer
import rouge
scorer = rouge.Rouge()
scores = scorer.get_scores(summ, summs[0])
print(scores)

reference='John really loves data science very much and studies it a lot.'
candidate='John very much loves data science and enjoys it a lot.'

print(scorer.get_scores(candidate, reference))