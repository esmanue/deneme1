import pandas as pd
import random
import pickle
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


stop_words = set(stopwords.words('english'))

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


df = pd.read_csv("merged_cleaned_dataset.csv")
X = df['clean_text']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

augmented_texts = []
augmented_labels = []

for text, label in zip(X_train, y_train):
    if label == 0:
        words = text.split()
        new_words = synonym_replacement(words, n=2)
        new_text = ' '.join(new_words)
        augmented_texts.append(new_text)
        augmented_labels.append(0)

X_train_all = pd.concat([X_train, pd.Series(augmented_texts)], ignore_index=True)
y_train_all = pd.concat([y_train, pd.Series(augmented_labels)], ignore_index=True)


vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train_all)
X_test_vec = vectorizer.transform(X_test)

model = LinearSVC(C=2, max_iter=10000)
model.fit(X_train_vec, y_train_all)

y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


