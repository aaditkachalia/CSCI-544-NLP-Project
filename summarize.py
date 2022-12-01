import re
import os
import nltk
import pickle
import pandas as pd 
from bs4 import BeautifulSoup

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import gensim

from nltk.tokenize import sent_tokenize

import torch
from transformers import BertTokenizer, BertModel

from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import json

import os
import nltk
import pickle
import pandas as pd 
from bs4 import BeautifulSoup

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import gensim

from flask import Response
from nltk.tokenize import sent_tokenize

import torch
from transformers import BertTokenizer, BertModel


app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
w2v_model = gensim.models.Word2Vec.load("./word2vec/word2vec.model")
device = 'cpu'
Bmodel = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased',
                                output_hidden_states = True, # Whether the model returns all hidden-states.
                                )
Bmodel.to(device)
Bmodel.eval()
nltk.download('punkt')


def preProcess(text):
    

    # sentences = sent_tokenize(text)
    sentences = text.split('.')[:-1]
    gold_sentences = []

    for sentence in sentences:
        
        if len(sentence.strip()) > 0:
            text = sentence.strip()
            text = text.strip('.')
            text = text.lstrip('0123456789.- ')
            text = re.sub(r'(( [\d])$)|(.*>)|(\n)','',text)
            text = text.strip()
            gold_sentences.append(text)
    
    return gold_sentences

def w2vEmbed(text):
    
    text = preProcess(text)
    sent_vector = []
    new_text = []

    for sentence in text:
        plus=0
        found = False
        for word in sentence.split():
            if word in w2v_model.wv.key_to_index:
                plus += w2v_model.wv[word]
                found = True
        if found:
            plus = plus/len(sentence.split()) 
            sent_vector.append(plus)
            new_text.append(sentence)
    return new_text, sent_vector

def bertEmbed(text):
    
    with torch.no_grad():

        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        outputs = Bmodel(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
    
    return sentence_embedding    

def legalBertEmbed(text):
    
    text = preProcess(text)
    sent_vector = []
    
    for sentence in text:
        embeddings = bertEmbed(sentence).cpu().numpy()
        sent_vector.append(embeddings)

    return text, sent_vector


def kmeansSummary(n_clusters, sent_vector, text):
    
    kmeans = KMeans(n_clusters, random_state = 42)
    y_kmeans = kmeans.fit_predict(sent_vector)
    summarization_list=[]
    
    for i in range(n_clusters):
        my_dict={}
        for j in range(len(y_kmeans)):
            if y_kmeans[j]==i:
                my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sent_vector[j])
        min_distance = min(my_dict.values())
        summarization_list.append(min(my_dict, key=my_dict.get)) 

    summary = []                         
    for i in sorted(summarization_list):
        # summary.extend([x.lower().strip() for x in text[i].split()])
        summary.append(text[i])
    summary = list(set(summary))
    
    return summary

def gmmSummary(n_clusters, sent_vector, text):

    gm = GaussianMixture(n_components=n_clusters, random_state=42).fit(sent_vector)
    y_prob = gm.predict_proba(sent_vector)
    y_cluster = gm.predict(sent_vector)
    form_cluster = {}
    
    for i in range(len(sent_vector)):
        if y_cluster[i] in form_cluster:
            form_cluster[y_cluster[i]].append([i, y_prob[i][y_cluster[i]]])
        else:
            form_cluster[y_cluster[i]] = [[i, y_prob[i][y_cluster[i]]]]
    
    summary = []
    best_indexes = []
    for i in range(n_clusters):
        cluster_val = form_cluster[i]
        cluster_val = sorted(cluster_val, key=lambda x:x[1], reverse=True)
        best_indexes.append(cluster_val[0][0])
    best_indexes.sort()
    
    for i in best_indexes:
        # summary.extend([x.lower().strip() for x in text[i].split()])
        summary.append(text[i])
    summary = list(set(summary))

    return summary


def w2vKmeans(text):

    try:
        text, sent_vector = w2vEmbed(text)
        n_clusters = max(1, len(text)//10)
        summary = kmeansSummary(n_clusters, sent_vector, text)
        response = {'status':200, 'summary':summary}
    
    except Exception as e:
        mini_summary = text[0:5]
        response = {'status':500, 'summary':mini_summary, 'error':e}
    
    return response


def w2vGMM(text):


    try:
        text, sent_vector = w2vEmbed(text)
        n_clusters = max(1, len(text)//10)
        summary = gmmSummary(n_clusters, sent_vector, text)
        response = {'status':200, 'summary':summary}

    except Exception as e:
        mini_summary = text[0:5]
        response = {'status':500, 'summary':mini_summary, 'error':e}
    
    return response


def bertKmeans(text):

    
    try:
        text, sent_vector = legalBertEmbed(text)
        n_clusters = max(1, len(text)//10)
        summary = kmeansSummary(n_clusters, sent_vector, text)
        response = {'status':200, 'summary':summary}
    
    except Exception as e:
        mini_summary = text[0:5]
        response = {'status':500, 'summary':mini_summary, 'error':e}
    
    return response

def bertGMM(text):

    try:
        text, sent_vector = legalBertEmbed(text)
        n_clusters = max(1, len(text)//10)
        summary = gmmSummary(n_clusters, sent_vector, text)
        response = {'status':200, 'summary':summary}
    
    except Exception as e:
        mini_summary = text[0:5]
        response = {'status':500, 'summary':mini_summary, 'error':e}
    
    return response


def unit_test(case):
    
    case = 'On September 3, 2020, US District Judge Robert N Scola sentenced Johnny Hidalgo to 100 months’ imprisonment and three years of supervised release for his supervision of the Everglades Peruvian call center to carry out a fraud and extortion scheme against American consumers.  As part of his guilty plea, Hidalgo admitted that he and others at Everglades in Peru called victims in the United States claiming to be attorneys or government representatives.  The callers falsely claimed that victims failed to pay for or receive certain product deliveries, and that victims owed thousands of dollars in fines as a result.  The callers threatened victims with negative credit reports, imprisonment, deportation, or property seizures that could be avoided only through the immediate payment of “settlement fees” to co-conspirators at the Angeluz Florida Corporation in Miami.  Hidalgo was arrested in 2016 by Peruvian authorities and extradited to the United States with two co-defendants last year.'

    print(w2vKmeans(case))
    print('-------------')
    print(w2vGMM(case))
    print('-------------')
    print(bertKmeans(case))
    print('-------------')
    print(bertGMM(case))
    print('-------------')


@app.route('/', methods=['GET'])
def index():
    title = 'Project Demo'
    return render_template('index.html',
                           title=title)

@app.route("/getdata", methods = ['POST'])
def getdata_func():
    jsondata = request.get_json()
    case = (jsondata['name']).split('\n')
    print(case[0])
    print(type(case[0]))
    example = ''
    for line in case:
        example += line.strip()+'.'
    #print(w2vKmeans(example))
    print("Calling w2vKmeans...")
    output1 = w2vKmeans(example)
    print("Calling w2vGMM...")
    output2 = w2vGMM(example)
    print("Calling bertKmeans...")
    output3 = bertKmeans(example)
    print("Calling bertGMM...")
    output4 = bertGMM(example)
    returnData=[]
    returnData.append({
            "r1":output1,
            "r2":output2,
            "r3":output3,
            "r4":output4
        })
    print('Output...')
    print(returnData)
    return {"data": returnData}
    # return "Hello, cross-origin-world!"

if __name__ == '__main__':
    app.run(debug=True)


#unit_test()
