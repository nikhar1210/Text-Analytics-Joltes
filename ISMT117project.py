#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:55:31 2019

@author: nikharshah
"""


import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
import os
from collections import Counter

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string

# Plotting tools
from sklearn.model_selection import train_test_split
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

os.chdir('/Users/nikharshah/FinalProject117')

# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())


df.head()

# Convert to list
data = df.content.values.tolist()
target = df.target.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])

CONTRACTION_MAP = {
    "ain`t": "is not",
    "aren`t": "are not",
    "can`t": "cannot",
    "can`t`ve": "cannot have",
    "`cause": "because",
    "could`ve": "could have",
    "couldn`t": "could not",
    "couldn`t`ve": "could not have",
    "didn`t": "did not",
    "doesn`t": "does not",
    "don`t": "do not",
    "hadn`t": "had not",
    "hadn`t`ve": "had not have",
    "hasn`t": "has not",
    "haven`t": "have not",
    "he`d": "he would",
    "he`d`ve": "he would have",
    "he`ll": "he will",
    "he`ll`ve": "he he will have",
    "he`s": "he is",
    "how`d": "how did",
    "how`d`y": "how do you",
    "how`ll": "how will",
    "how`s": "how is",
    "I`d": "I would",
    "I`d`ve": "I would have",
    "I`ll": "I will",
    "I`ll`ve": "I will have",
    "I`m": "I am",
    "I`ve": "I have",
    "i`d": "i would",
    "i`d`ve": "i would have",
    "i`ll": "i will",
    "i`ll`ve": "i will have",
    "i`m": "i am",
    "i`ve": "i have",
    "isn`t": "is not",
    "it`d": "it would",
    "it`d`ve": "it would have",
    "it`ll": "it will",
    "it`ll`ve": "it will have",
    "it`s": "it is",
    "let`s": "let us",
    "ma`am": "madam",
    "mayn`t": "may not",
    "might`ve": "might have",
    "mightn`t": "might not",
    "mightn`t`ve": "might not have",
    "must`ve": "must have",
    "mustn`t": "must not",
    "mustn`t`ve": "must not have",
    "needn`t": "need not",
    "needn`t`ve": "need not have",
    "o`clock": "of the clock",
    "oughtn`t": "ought not",
    "oughtn`t`ve": "ought not have",
    "shan`t": "shall not",
    "sha`n`t": "shall not",
    "shan`t`ve": "shall not have",
    "she`d": "she would",
    "she`d`ve": "she would have",
    "she`ll": "she will",
    "she`ll`ve": "she will have",
    "she`s": "she is",
    "should`ve": "should have",
    "shouldn`t": "should not",
    "shouldn`t`ve": "should not have",
    "so`ve": "so have",
    "so`s": "so as",
    "that`d": "that would",
    "that`d`ve": "that would have",
    "that`s": "that is",
    "there`d": "there would",
    "there`d`ve": "there would have",
    "there`s": "there is",
    "they`d": "they would",
    "they`d`ve": "they would have",
    "they`ll": "they will",
    "they`ll`ve": "they will have",
    "they`re": "they are",
    "they`ve": "they have",
    "to`ve": "to have",
    "wasn`t": "was not",
    "we`d": "we would",
    "we`d`ve": "we would have",
    "we`ll": "we will",
    "we`ll`ve": "we will have",
    "we`re": "we are",
    "we`ve": "we have",
    "weren`t": "were not",
    "what`ll": "what will",
    "what`ll`ve": "what will have",
    "what`re": "what are",
    "what`s": "what is",
    "what`ve": "what have",
    "when`s": "when is",
    "when`ve": "when have",
    "where`d": "where did",
    "where`s": "where is",
    "where`ve": "where have",
    "who`ll": "who will",
    "who`ll`ve": "who will have",
    "who`s": "who is",
    "who`ve": "who have",
    "why`s": "why is",
    "why`ve": "why have",
    "will`ve": "will have",
    "won`t": "will not",
    "won`t`ve": "will not have",
    "would`ve": "would have",
    "wouldn`t": "would not",
    "wouldn`t`ve": "would not have",
    "y`all": "you all",
    "y`all`d": "you all would",
    "y`all`d`ve": "you all would have",
    "y`all`re": "you all are",
    "y`all`ve": "you all have",
    "you`d": "you would",
    "you`d`ve": "you would have",
    "you`ll": "you will",
    "you`ll`ve": "you will have",
    "you`re": "you are",
    "you`ve": "you have"
                    }

stopword_list = nltk.corpus.stopwords.words('english')

stopword_list.append('as','like', 'subject', 'line', 'say', 'know', 'people', 'dont', 'make', 'think')

def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens


def expand_contractions(text, contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None


# lemmatize text based on POS tags  
def lemmatize_text(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:            
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


#def unescape_html(parser, text):
#
#    return html.unescape(text)



def normalize_corpus(corpus, lemmatize=True, tokenize=False):

    normalized_corpus = []  
    for text in corpus:
        text = text.lower()
        text = expand_contractions(text, CONTRACTION_MAP)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus



normalized_data = normalize_corpus(data,lemmatize=True)

normalized_train, normalized_test, target_train, target_test = train_test_split(normalized_data, target, test_size=0.20, random_state=42)

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=20,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             ngram_range=(1,2)                  # ngram <= 2
                            )

data_vectorized = vectorizer.fit_transform(normalized_train)

# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


word_dictionary = pd.DataFrame(list(vectorizer.vocabulary_.items()), columns = ['Word', 'Location'])

word_dictionary = word_dictionary.set_index('Location')

word_dictionary = word_dictionary.sort_index()

word_list = word_dictionary['Word'].tolist()

for i in range(len(word_list)-1,-1,-1): 
    a = Counter(word_list[i].split())    
    if (a.most_common()[0][1] > 1 and len(a.keys()) <= 1):
        del word_list[i]
        
for i in range(len(word_list)-1,-1,-1): 
    if len(word_list[i]) <= 2:
        del word_list[i]

vectorizer = CountVectorizer(stop_words='english', ngram_range= (1,2), vocabulary = word_list)

counts = data_vectorized.sum(axis=0).A1

freq_distribution = Counter(dict(zip(word_list, counts)))
freq_distribution = pd.DataFrame(freq_distribution.most_common())
freq_distribution.to_csv('word_freq.csv')

set(df['target_names'])

data_vectorized = vectorizer.fit_transform(normalized_train)

lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )

lda_output = lda_model.fit_transform(data_vectorized)

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))


pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(panel, 'model.html')


#Supervised Learning 
# Predict the topic
mytext = ["Two former close aides to Prime Minister Benjamin Netanyahu of Israel are among those facing likely criminal charges in one of the country’s biggest corruption scandals yet, arising from the multibillion-dollar purchase of submarines and missile boats from Germany. A former chief of staff to Mr. Netanyahu, David Sharan, is accused of accepting bribes from an Israel agent for the shipbuilder, ThyssenKrupp Marine Systems. David Shimron, who has served as a personal lawyer and close confidant to Mr. Netanyahu and is also his second cousin, is accused of laundering money to help the shipbuilders agent, Michael Ganor, conceal his role in a separate financial transaction"]

mytext = ["Three separate attacks targeted busy marketplaces in northwest Syria, producing a brutal death toll Monday in the war-torn area, according to local volunteer and rights groups. At least 19 people were killed, 12 of them children. In the province of Idlib, Syria's last major opposition bastion, airstrikes ripped through two marketplaces, killing at least nine people in Maarat al-Numan and nearby Saraqeb, said a volunteer rescue group known as the White Helmets. At least four of the dead in the attack on Maarat al-Numan were children, according to the Syrian American Medical Society. Footage released by the White Helmets showed members carrying bodies away from the scene in the town of Maarat al-Numan, where lettuce and onions were strewn on the blood-stained ground. Another video from the same town shared by the White Helmets pictured a young girl, her face covered in blood, crying next to her brother and calling for their father.Please tell me my dad is still alive, please be alive my dad,she says."]


norm_text = vectorizer.transform(normalize_corpus(mytext))

norm_text = vectorizer.transform(normalized_test)

#Predicted Model Output
model_output = lda_model.transform(norm_text)



# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(normalized_test))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=topicnames, index=docnames)


# Get dominant topic for each document
dominant_topic = []
for i in range(len(df_document_topic)):
       dominant_topic.append(np.argsort(df_document_topic.values[i])[::-1][0:1])
dominant_topic = np.asarray(dominant_topic)# df_document_topic['dominant_topic'] = dominant_topic

#Dominant Topic Weight
dominant_topic_wt = []
for i in range(len(df_document_topic)):
       dominant_topic_wt.append(np.sort(df_document_topic.values[i])[-1])
dominant_topic_wt = np.asarray(dominant_topic_wt)

#Second Dominant Topic
second_dominant_topic = []
for i in range(len(df_document_topic)):
       second_dominant_topic.append(np.argsort(df_document_topic.values[i])[::-1][1:2])
second_dominant_topic = np.asarray(second_dominant_topic)

#2nd Dominant Topic Weight
second_dominant_topic_wt = []
for i in range(len(df_document_topic)):
       second_dominant_topic_wt.append(np.sort(df_document_topic.values[i])[-2])
second_dominant_topic_wt = np.asarray(second_dominant_topic_wt)


#Third Dominant Topic
third_dominant_topic = []
for i in range(len(df_document_topic)):
       third_dominant_topic.append(np.argsort(df_document_topic.values[i])[::-1][2:3])
third_dominant_topic = np.asarray(third_dominant_topic)


#3rd Dominant Topic Weight
third_dominant_topic_wt = []
for i in range(len(df_document_topic)):
       third_dominant_topic_wt.append(np.sort(df_document_topic.values[i])[-3])
third_dominant_topic_wt = np.asarray(third_dominant_topic_wt)



df_document_topic["Actual Group"] = target_test
df_document_topic['dominant_topic'] = dominant_topic + 1
df_document_topic['dominant_topic_wt'] = dominant_topic_wt
df_document_topic['second_dominant_topic'] = second_dominant_topic + 1 
df_document_topic['second_dominant_topic_wt'] = second_dominant_topic_wt
df_document_topic['third_dominant_topic'] = third_dominant_topic + 1
df_document_topic['third_dominant_topic_wt'] = third_dominant_topic_wt



####Supervised Model

from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier

newsgroups_test = fetch_20newsgroups(subset='test')
text_test = newsgroups_test['data']
target_test = newsgroups_test['target']

normalized_test_data = normalize_corpus(text_test, lemmatize=True)

test_data_vectorized = vectorizer.fit_transform(normalized_test_data)

X_train = data_vectorized

Y_train = target

##Randomforest Model

clf = RandomForestClassifier(n_estimators=1000, random_state=0)

clf.fit(X_train,Y_train)

#Predicting Output
test_data_pred = clf.predict(test_data_vectorized)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

columns_name = ('atheism',
'computer.graphics',
'os.mswindows.miscellaneous',
'computersystems.ibm.pchardware',
'computer.systems.mac.hardware',
'windows.x',
'miscellaneous.forsale',
'Autos',
'motorcycles',
'sports.baseball',
'sports.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'religion.christian',
'politcs.guns',
'politics.middleeast',
'politics.miscellaneous',
'religion.miscellaneous')

Confusion_Matrix = pd.DataFrame(confusion_matrix(target_test,test_data_pred), index = row_names)




