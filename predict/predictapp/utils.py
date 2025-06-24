import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

def clean_data(data):
    data = data.lower()
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    stop_words = set(stopwords.words('english'))
    not_stopword = ["not", "no", "never", "doesn't", "isn't", "don't", "wasn't", "won't", "didn't", "can't"]
    stop_words = stop_words - set(not_stopword)
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(data)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

def clean_data1(data):
    stop_words = set(stopwords.words('english'))
    data = data.lower()
    data = re.sub(r'\s+', ' ', data).strip() #sağ ve soldaki ve ortadaki bosluklari kaldirir
    data = data.split()
    filtered_words = [word for word in data if word not in stop_words] #stopwordsleri filtreliyor
    return ' '.join(filtered_words)

