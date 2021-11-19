import tkinter
import nltk
import pickle
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from lightgbm import LGBMClassifier


prediction_model = pickle.load(open('finalized_model.sav', 'rb'))
token_model      = pickle.load(open('token_model.sav', 'rb'))


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


# Let's create the Tkinter app
app = tkinter.Tk()
app.geometry("1600x800")
app.title("Graphical User Interface")
label = tkinter.Label(app, text = "Опишите вашу проблему в нижнем окне:").place(relx = 0.5, rely= 0.07, anchor='center')

# creating a function called DataCamp_Tutorial()
question = tkinter.StringVar()


def callbackFunc():
    
    prep_question  = preprocess_sentence(question.get())
    token_question = token_model.transform([prep_question])
    print(token_question.toarray())
    tokens = token_question.toarray()
    
    label = prediction_model.predict(tokens)
    
    switcher={
                0:'пульмонологу',
                1:'аллергологу',
                2:'кардиологу',
                3:'ковид специалисту',
                4:'дерматологу'
    }
    
    result = switcher.get(label[0],"...")
    
    tkinter.Label(app, text = "Вам нужно обратиться к {}.".format(result)).place(relx = 0.5, rely= 0.85, anchor='center')

entry = tkinter.Entry(app, textvariable=question, justify="center")
entry.place(relx = 0.5, rely= 0.4, anchor='center', width=1000, height=400)
        
resultButton = tkinter.Button(app, text = 'Получить ответ',
                         command=callbackFunc).place(relx = 0.5, rely= 0.75, anchor='center')


app.mainloop()
