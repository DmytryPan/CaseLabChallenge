from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib

import re
# Create your views here.

model = joblib.load('classifier/models/RandomForestClassifier.joblib')
vectorizer = joblib.load('classifier/models/rf_vectorizer.joblib')

stop_words = set(stopwords.words('english'))

stemmer = snowball.SnowballStemmer('english')

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text)) + ' ' + ' '.join(emoticons).replace('-', '')
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    text = ' '.join(tokens)
    return text

def classify_review(request):
    if request.method =='POST':
        review = preprocess(request.POST.get('review')) # Извлекли текст ревью из POST запроса и предобработали
        transformed_review = vectorizer.transform([review]) # Векторизация переданного текста 
        pred = model.predict(transformed_review)
        probs = model.predict_proba(transformed_review)

        negative_prob, positive_prob = probs[0], probs[1]

        rating =  5 + 5*positive_prob if pred == 1 else 5 * (1 - negative_prob)
        
        result = {
            'prediction': 'positive' if pred == 1 else 'negative',
            'rating' : round(rating, 1) 
        }
        return JsonResponse(result)
    return render(request, 'classifier/templates/classify_review.html')
        



