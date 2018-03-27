import nltk
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from pymystem3 import Mystem

def count(arr):
    D = {}
    for i in arr:
        if i in D:
            D[i] += 1
        else:
            D[i] = 1
    return D

punct = '.,!?()/\\:;«»–'

with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
words = text.split()
words = [word.strip(punct) for word in words]

words[:100]
clean_text = ''
for word in words:
    clean_text += word + ' '

m = Mystem()
lemmas = ['']
for word in m.lemmatize(clean_text):
    if len(word)>2:
        lemmas.append(word)
for lemma in lemmas:
    if len(lemma) <= 1:
        lemmas.remove(lemma)


lemmas1 = lemmas[:700]
lemmas2 = lemmas[701:1400]
lemmas3 = lemmas[1401:2100]
lemmas4 = lemmas[2101:]

lemD1 = sorted(count(lemmas1).items(), key=lambda x: x[1])[len(count(lemmas1))-5:]
lemD2 = sorted(count(lemmas2).items(), key=lambda x: x[1])[len(count(lemmas2))-5:]
lemD3 = sorted(count(lemmas3).items(), key=lambda x: x[1])[len(count(lemmas3))-5:]
lemD4 = sorted(count(lemmas4).items(), key=lambda x: x[1])[len(count(lemmas4))-5:]

newSeed = []

for i in lemD1:
    if i[0] not in newSeed:
        newSeed.append(i[0])
for i in lemD2:
    if i[0] not in newSeed:
        newSeed.append(i[0])
for i in lemD3:
    if i[0] not in newSeed:
        newSeed.append(i[0])
for i in lemD4:
    if i[0] not in newSeed:
        newSeed.append(i[0])

print (newSeed)

    
