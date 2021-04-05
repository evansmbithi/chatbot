#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# # Install NLTK packages

# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('popular', quiet=True) #downloads packages
#nltk.download('punkt') #first-time use only
#nltk.download('wordnet') #first time


# # Reading the corpus

# In[3]:


f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower() #converts text to lowercase


# # Tokenization

# In[4]:


import nltk
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


# # Preprocessing

# In[5]:


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# # Keyword Matching

# In[6]:


GREETING_INPUTS = ("hello", "hi", "greetings", "ssup", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "'nods'", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# # Generating response

# In[7]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry, I did not understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


# In[ ]:


flag=True
print("EV: My name is EV. I will answer your queries about chatbots. If you wish to exit, type Bye!")
while(flag==True):    
    user_response = input('You: ')
    user_response=user_response.lower()
    if(user_response!='Bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("EV: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("EV: "+greeting(user_response))
            else:
                print("EV: ",end='')
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("EV: Bye! Take care..")


# In[ ]:




