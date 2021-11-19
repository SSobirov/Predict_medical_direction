import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import numpy as np

def preprocess_data(data):
    filtered_tokens = []
    snowball = SnowballStemmer(language="russian") 
    for row in data.X:
        filtered_sentence = ""
        text = row.lower()
        text_p = "".join([char for char in text if char not in string.punctuation])
        words = word_tokenize(text_p, language="russian")
        stop_words = stopwords.words('russian')
        stop_words.extend(["здравствуйте","подскажите","пожалуйста", "очень", "помогите", "спасибо","это","эта", "года","лет","месяц","день","врач","1","2","3","4","5","6","7","8","9","делать"])
        for token in words:
            if token not in stop_words:
                filtered_sentence = filtered_sentence + snowball.stem(token) + " "
                #filtered_tokens.append(snowball.stem(token))
        filtered_tokens.append(filtered_sentence)      
        
    return filtered_tokens


def preprocess_sentence(sentence):
    snowball = SnowballStemmer(language="russian") 
    text = sentence.lower()
    text_p = "".join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text_p, language="russian")
    filtered_sentence = ""
    for token in words:
        stemmed_word = snowball.stem(token)
        filtered_sentence = filtered_sentence + stemmed_word + " " 
    return filtered_sentence



Allergist_data     = pd.read_csv("../data/allergist.csv")
Cardiologist_data  = pd.read_csv("../data/cardiologist.csv")
Covid_data         = pd.read_csv("../data/covid.csv")
Dermatologist_data = pd.read_csv("../data/dermatologist.csv")
Pulmonologist_data = pd.read_csv("../data/pulmonologist.csv")   

Allergist_tokens     = preprocess_data(Allergist_data)
Cardiologist_tokens  = preprocess_data(Cardiologist_data)
Covid_tokens         = preprocess_data(Covid_data)
Dermatologist_tokens = preprocess_data(Dermatologist_data)
Pulmonologist_tokens = preprocess_data(Pulmonologist_data)
final = Pulmonologist_tokens + Allergist_tokens + Cardiologist_tokens + Covid_tokens + Dermatologist_tokens

"""
Allergist_freq     =  nltk.FreqDist(Allergist_tokens)
Cardiologist_freq  =  nltk.FreqDist(Cardiologist_tokens)
Covid_freq         =  nltk.FreqDist(Covid_tokens)
Dermatologist_freq =  nltk.FreqDist(Dermatologist_tokens)
Pulmonologist_freq =  nltk.FreqDist(Pulmonologist_tokens)


main_words = []

for word in Allergist_freq.most_common(300):
    main_words.append(word[0])
for word in Cardiologist_freq.most_common(300):
    main_words.append(word[0])
for word in Covid_freq.most_common(300):
    main_words.append(word[0])
for word in Dermatologist_freq.most_common(300):
    main_words.append(word[0])
for word in Pulmonologist_freq.most_common(300):
    main_words.append(word[0])
    
"""
    
     
CountVec = CountVectorizer(ngram_range=(1,1),max_features=707)
#CountVec = TfidfVectorizer(analyzer='word')
Count_data = CountVec.fit_transform(final)

cv_dataframe=pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
#print(cv_dataframe)
labels = pd.read_excel('../data/labels.xlsx')
cv_dataframe['Y'] = labels.Y
writer = pd.ExcelWriter('../training/prepared_data.xlsx', engine='openpyxl')
cv_dataframe.to_excel(writer, index=False)    
writer.save()  

