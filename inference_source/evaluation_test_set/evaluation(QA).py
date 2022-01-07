#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install contractions


# In[4]:


import numpy as np
import pandas as pd
import spacy
import re
import nltk
import contractions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
from transformers import pipeline


# In[5]:


get_ipython().system('pip install openpyxl')


# In[6]:


test=pd.read_excel('../input/enterpret-absa/test.xlsx')


# In[ ]:


#nlp = spacy.load('en_core_web_lg')


# In[7]:


def preprocess(text):
    #to lowercase
    text = str(text).lower()
    #remove urls
   
    #remove emails
    text = re.sub(r'\S*@\S*\s?',' ',text)
    #remove mentions
    text = re.sub(r'@\S+', ' ', text)
    #contractions
    text = contractions.fix(text)
    #remove hashtags
    text = re.sub(r'@\S+', ' ', text)
    #remove emojis
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    #remove all punct
    text = re.sub('[^A-z0-9]', ' ', text)
    #remove extras whitespaces
    text = re.sub(' +', ' ', text)
    return text


# In[8]:


#loading the Question answering model
qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_model = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)


# In[9]:


#loading the sentiment analysis model
sent_tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
sent_model = pipeline('sentiment-analysis',model="finiteautomata/bertweet-base-sentiment-analysis",tokenizer=sent_tokenizer)


# In[20]:


def get_sentiment(aspect,text,qa_model,sent_model):
    
    question = f'how is {aspect}'#question from aspect
    QA_input = {'question': question, 'context': text}
    qa_result = qa_model(QA_input)
    answer = qa_result['answer']#qa model result
    #print(question)
    #print(answer)
      #sentiment model 
    sent_result = sent_model(answer)    
    print(sent_result)
    sentiment = sent_result[0]['label']
    #print(sentiment)

    if sentiment == 'NEG':
        sentiment, score = 'Negative', 0
    elif sentiment == 'NEU':
        sentiment, score = 'Neutral', 1
    else:
        sentiment, score = 'Positive', 2
    

    return sentiment,score


# In[21]:


def compute(text, aspect, qa_model, sent_model):
    text = preprocess(text)
    #noun_list = get_noun(preprocess_text)
    #aspect_classes = get_similar_words(noun_list, aspects)
   # print(aspect_classes)
    #print(noun_list)
    aspect=preprocess(aspect)
    sentiment_result ,score= get_sentiment(aspect, text, qa_model, sent_model)
    return sentiment_result,score
    
   
    


# In[22]:


ans=[]
#inference on test set
for i in range(len(test)):
    text=test.loc[i,'text']
    aspect=test.loc[i,'aspect']
    sentiment,lab=compute(text,aspect,qa_model,sent_model)
    ans.extend([lab])
label=pd.Series(ans)
    


# In[23]:


data=test.copy()


# In[24]:


data['label']=label

data.to_csv('aq_sent_.csv')


# In[ ]:




