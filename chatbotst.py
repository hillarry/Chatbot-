# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:10:30 2022

@author: User
"""

import streamlit as st
import os
import sys
import requests

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import pickle
import numpy as np

#from bot import Bot

#from tensorflow import keras
from tensorflow.keras.models import load_model
model=load_model('chatbot_model.h5')
import json
import random
intents=json.loads(open('ques4.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence,words,show_details=True):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i]=1
                if show_details:
                    print("found in bag: %s" %w)
    return(np.array(bag))
    
def predict_class(sentence,model):
    p=bow(sentence,words,show_details=False)
    res=model.predict(np.array([p]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    results.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag=ints[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result=random.choice(i['responses'])
    return result

def chatbot_response(text):
    ints=predict_class(text,model)
    res=getResponse(ints,intents)
    return res

st.sidebar.title("NLP bot")

st.title("""
Depato 
This chatbot is intended for answering the questions of Information Technology Department.You can ask anything about our department.
""")

st.title("Depato Bot is Ready to Help you!!!")


def get_text():
    input_text = st.text_input("You: ")
    return input_text 
if st.sidebar.button('Initialize bot'): 
    user_input = get_text()

#st.text_input("Talk to the bot",key='input_text',on_change=chatbot_response(user_input))
if user_input:
    st.text_area("Bot:",chatbot_response(user_input))
    
