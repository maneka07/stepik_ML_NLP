# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:17:32 2022

This script uses the TextRank algorithm to create a summary 
of a news article by choosing top sentences from the ranking.
The generated summary is then compared with two model summaries:
the first one is the summary taken from the news website, the second
one is the first few sentences from the article.

@author: Tatiana
"""

import io
import csv
import nltk
import string
import numpy as np
from copy import deepcopy

def tokenize_me(text, lang='russian', stemmer=None):
    '''
    Parameters
    ----------
    text : string
        text to tokenize.
    lang : string, optional
        Language of the text.
    stemmer: optional
        Word stemmer. No stemmer is used by default.
    Returns
    -------
    list of tokens.
    '''
    table = str.maketrans('','', string.punctuation)
    stopw = set(nltk.corpus.stopwords.words(lang))
    toks = nltk.tokenize.word_tokenize(text)
    if stemmer == None:
        toks = [w.translate(table).lower() for w in toks if len(w) > 2 and w not in stopw]
    else:
        toks = [stemmer.stem(w.translate(table).lower()) for w in toks if len(w) > 2 and w not in stopw]
    return toks

def text_rank(text, stemmer=None, lang='russian', p_tele=0.15, max_iter=200):
    '''
    Parameters
    ----------
    text : string
        Text whose sentences to rank.
    stemmer : optional
        Specify a stemmer. The default is None.
    lang: string, optional
        Language of the text.
    p_tele : float, optional
        Teleportation probability for PageRank algorithm. The default is 0.15.
    max_iter : int, optional
        Maximum number of iteration for the PageRank algorithm.
    Returns
    -------
    A list of tuples (sentence_id, rank).
    '''
    sents = nltk.tokenize.sent_tokenize(text, lang) 
    sent_toks = []
    for s in sents:
        if len(s) > 1:
            sent_toks.append(tokenize_me(s, lang, stemmer))
       
    #count common words in two sentences
    mat = np.zeros((len(sent_toks), len(sent_toks)))
    for i, s1 in enumerate(sent_toks):
        for j, s2 in enumerate(sent_toks[i+1:]):
            cnt = len([w for w in s1 if w in s2])
            mat[i][i+1+j] = mat[i+1+j][i] = cnt
    #transition probabilities
    mat = [[x/sum(y) if sum(y) > 0 else 1/len(y) for x in y] for y in mat]
    #add probability of teleportation
    mat =[[ x*(1-p_tele) + p_tele/len(mat) for x in y] for y in mat]
    mat = np.array(mat)
    
    state = np.zeros(len(sent_toks))
    state[0] = 1.0
    for i in range(max_iter): 
        old_state = deepcopy(state)
        state = mat.transpose(1,0).dot(state)
        #print(state-old_state)
        #print(i, (state-old_state).sum())
        #print(old_state, state)
        if (state == old_state).all():
            #print ("converged at ", i)
            break
    sent_rank = sorted([(k,v) for k, v in enumerate(state)], key=lambda x: x[1], reverse=True)
    return sent_rank


def rouge_score(summ, baseline, lang='russian', nval=1):
    '''
    Parameters
    ----------
    summ : string
        Generated summary that we want to score.
    baseline : string
        Baseline summary against which we score the generated
        summary.
    nval : int
    N-value for N-gramms
    Returns
    -------
    float number computed by the formula:
    (number of n-grams in model and reference) / 
    (number of n-grams in model)

    '''
    assert nval in {1}
    if nval == 1:
        toks_summ = tokenize_me(summ, lang=lang)
        toks_base = tokenize_me(baseline, lang=lang)
    a = [t for t in toks_summ if t in toks_base]
    #print(len(a), len(toks_base))
    #print(a, '\n', toks_base)
    return len(a)/len(toks_base)


texts = []
summs = []
with io.open('data/news.csv', encoding='utf-8',newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        texts.append(row[0])
        summs.append(row[1])
        
print(f"Will process {len(texts)} news and their summaries")


np.set_printoptions(precision=10)
#stemmer = nltk.stem.porter.PorterStemmer()
lang = "russian"
#teleportation probability
p_tele = 0.3
#stop adding sentences to the summary when the length exceeds max_summ_len symbols
max_summ_len = 300 
for i, text in enumerate(texts):
    ranks = text_rank(text, lang=lang, p_tele=p_tele,max_iter=400)
    sents = nltk.tokenize.sent_tokenize(text, language=lang)
    sents = [s for s in sents if len(s) > 1]
    summ = "" 
    for i in range (len(sents)):
        summ += sents[ranks[i][0]]
        if len(summ) >= max_summ_len:
            break
    rouge1_1 = rouge_score(summ[0:max_summ_len], summs[i])
    rouge1_2 = rouge_score(summ[0:max_summ_len], text[0:max_summ_len])
    print(f"Rouge score for model 1 {rouge1_1}, model 2 {rouge1_2}")
# print(f"Generated summary: {summ}")
# print("---------------")
# print(f"baseline 1:{summs[4]}")
# print("---------------")
# print(f"baseline 2:{text[0:max_summ_len]}")




