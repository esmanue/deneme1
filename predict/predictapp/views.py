from rest_framework.views import APIView
from rest_framework.response import Response
from gensim.utils import simple_preprocess
import pickle
import os
from .utils import clean_data
from .utils import clean_data1
import pandas as pd
import re
import numpy as np
from gensim import corpora, models
from nltk.tokenize import sent_tokenize
from transformers import pipeline 
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #for base file path
model_path = os.path.join(BASE_DIR, 'predict', 'classifier.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'predict', 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


def predictions(text):
    cleaned = clean_data(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

def calculateCharacter(value):
    min_value=20*value
    max_value=40*value
    return{
        "max": max_value,
        "min": min_value
    }

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def Normalization(text,sentence_number):
    sentence_number=int(sentence_number)
    text=str(text)
    text = clean_data1(text)
    sentences = sent_tokenize(text) #tokenize text to sentences 
    tokenized = []
    for sentence in sentences:
        tokenized.append(simple_preprocess(sentence, deacc=True)) #for tokenize and deacc= removes accent marks from characters
    my_dictionary = corpora.Dictionary(tokenized) #to  create dictionary
    BoW_corpus = [my_dictionary.doc2bow(doc, allow_update=True) for doc in tokenized] #doc2bow increase freq

    tfIdf = models.TfidfModel(BoW_corpus, smartirs='ntc') #ntc = natural freq 
    weight_tfidf = []
    for doc in tfIdf[BoW_corpus]:
        for id, freq in doc:
            weight_tfidf.append([my_dictionary[id], np.around(freq, decimals=3)]) #around() returns a new array with each element rounded to the given number of decimals.
    
    weight_tfidf.sort(key=lambda x: x[1], reverse=True)

    control = [] # we have control list to control words 
    clear_list = []
    for vocab, freq in weight_tfidf:
        if vocab not in control: #if we dont have in control list we can add the kelime in the clear_list
            clear_list.append([vocab, freq]) 
            control.append(vocab)

    sentence_scores = []
    for sentence in sentences:
        words = simple_preprocess(sentence, deacc=True) #again tokenize
        score = 0 #count for sentences tdidf scores
        for word in words:
            for vocab, tfidf in clear_list: 
                if word == vocab:
                    score += tfidf 
        sentence_scores.append((sentence, round(score, 3)))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    sorted_summary = sorted(sentence_scores[:sentence_number],key=lambda x: sentences.index(x[0]))

    sorted_summary = [sentence for sentence, _ in sorted_summary]
    print("article:")
    print(f"{text}")

    print("\nsummary:")
    print(sorted_summary)
    
    
    calculateCharacterVal=calculateCharacter(sentence_number)
    print(calculateCharacterVal)
    summary = summarizer(text, max_length=calculateCharacterVal["max"], min_length=calculateCharacterVal["min"], do_sample=False)
    print(summary[0]['summary_text'])
    summarize=str(summary[0]['summary_text'])

    return {
        "tfidf": sorted_summary,
        "HuggFace": summarize
    }


class SummarizeNormalize(APIView):
    def post(self, request):
        text = request.data.get("text")
        sentence_number = request.data.get("sentence_number")
        result = Normalization(text,sentence_number)
        return Response({
            "tfidf": result["tfidf"],
            "HuggFace": result["HuggFace"]
            
        })

class PredictOverview(APIView):
    def post(self, request):
        text = request.data.get("text")
        sentence_number = request.data.get("sentence_number")
        prediction = predictions(text)
        result = Normalization(text,sentence_number)
        return Response({"prediction": int(prediction),
                         "tfidf": result["tfidf"],
                          "HuggFace": result["HuggFace"]
                         })

class CorrectLabel(APIView):
    def post(self, request):
        text = request.data.get("text")
        label = int(request.data.get("label"))
        prediction = predictions(text)

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        merge_path = os.path.join(BASE_DIR, 'predict', 'merged_cleaned_dataset.csv')
        compare_path = os.path.join(BASE_DIR, 'predict', 'compare.csv')

        df = pd.read_csv(merge_path, names=["Text", "label","clean_text"])
        text = re.sub(r'\s+', ' ', text).strip() #ortada bırakılan bosluklar icin stripe re ekledim.
        
        if text in df["Text"].values: #textin içinde arama yapıyor
            return Response({'message': 'Bu kayıt zaten mevcut.'} )
        
        else:
            cleantext=clean_data(text)
            with open(merge_path, "a", encoding="utf-8") as f:
                f.write(f'"{text}",{label},"{cleantext}"\n')

            if prediction != label:
             with open(compare_path, "a", encoding="utf-8") as f:
                f.write(f'"{text}","{prediction}","{label}"\n')

        return Response({'message': 'Feedback kaydedildi.'})