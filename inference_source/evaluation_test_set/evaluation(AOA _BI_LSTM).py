#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:





# This inference file may give errors if words not included in either the train or test set are given.

# In[ ]:


import pandas as pd #Import the libraries
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import os
import urllib.request
import tensorflow as tf
import torch
import os
import torchvision
import tarfile
from torch.utils.data import random_split
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
import torch.nn as nn
import spacy
from torch.utils.data import DataLoader


# In[ ]:


from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
from transformers import BertTokenizer
from torch.utils.data import Dataset
#pretrained_bert="bert-base-uncased"
import json


# In[ ]:


pip install openpyxl


# In[ ]:


test=pd.read_excel('../input/enterpret-absa/test.xlsx')


# In[ ]:


def parse_data(data):
    for i in range(len(data)):
        if(type(data.loc[i,'aspect'])==float):
            data.loc[i,'aspect']=str(data.loc[i,'aspect'])
    data['text_tok'] = data['text'].apply(lambda x: x.lower())
    data['text_tok'] = data['text_tok'].apply(custom_tokenize)
    data['aspect_tok'] = data['aspect'].apply(lambda x: x.lower())
    data['aspect_tok'] = data['aspect_tok'].apply(custom_tokenize)
    return data

def custom_tokenize(text):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    words = [word for word in tokens if word.isalnum()]
    return words

def max_len(data):
    max_text=0
    max_asp=0
    for i in range(len(data)):
        max_text=max(max_text,len(data.loc[i,'text_tok']))
        max_asp=max(max_asp,len(data.loc[i,'aspect_tok']))
    return max_text,max_asp

def find_w(data_train_1):
    word2idx={}
    i=0
    for sentence in data_train_1['text_tok']:
        for word in sentence:
            if word not in word2idx.values():
                word2idx[i]= word
                i=i+1
    for sentence in data_train_1['aspect_tok']:
        for word in sentence:
            if word not in word2idx.values():
                word2idx[i]= word
                i=i+1
                
    return word2idx
    


def create_vocab(data_train_1):
    word2idx = {}
    
    for sentence in data_train_1['text_tok']:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    for sentence in data_train_1['aspect_tok']:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    idx2word ={v: k for k, v in word2idx.items()}
    return word2idx,idx2word

def create_dict(vocab,embed):
    max_len=len(vocab)
    w_ind={}
    i_w={}
    print(max_len)
    for i in range(len(vocab)):
        if vocab[str(i)] not in w_ind:
            w_ind[vocab[str(i)]]=len(w_ind)+1
            i_w[i+1]=vocab[str(i)]
    
    return   w_ind,i_w    


# In[ ]:


test2=test.copy()
data=parse_data(test2)

#print(data)
max_tex,max_asp=max_len(data)


# In[ ]:


data


# Loading the vocab file which stored words to index dictionary

# In[ ]:


vocab_file= open("../input/lstm-vocab/full_vocabulary.json")
vocab_f=vocab_file.read()


# In[ ]:



vocab=json.loads(vocab_f)
print(len(vocab))
#w_ind,i_w=create_dict(vocab,300)


# In[ ]:


w_ind,i_w=create_dict(vocab,300)


# In[ ]:


#i_w


# In[ ]:


#vocab.keys()


# In[ ]:


max_tex=278#max sentence length
max_asp=8#max aspect length


# In[ ]:


tex=np.empty([len(data),max_tex])
aspec=np.empty([len(data),max_asp])
for i in range(len(data)):#creating a matrix of glove embeddings of all words in test
    #print(i)
    sent=data.loc[i,'text_tok']
    asp=data.loc[i,'aspect_tok']
    encoded_sentences = np.array([w_ind[word] for word in sent])
    if len(encoded_sentences)>278:
        encoded_sentences=encoded_sentences[:278]
    encoded_sen=np.append(encoded_sentences,np.zeros(max_tex-len(encoded_sentences)))
    encoded_asp=np.array([w_ind[word] for word in asp])
    encoded_as=np.append(encoded_asp,np.zeros(max_asp-len(encoded_asp)))
    tex[i]=encoded_sen
    aspec[i]=encoded_as


# In[ ]:


#data.loc[284]


# In[ ]:


tex=torch.LongTensor(tex)
aspec=torch.LongTensor(aspec)


# In[ ]:


test_ds=TensorDataset(tex,aspec)
batch_size = 20 #put batch size=20 for LSTM_AOA and 2 for LSTM_imp
test_dl = DataLoader(test_ds, batch_size, shuffle=False)#dataloader creation


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


class aspectClassificationBase(nn.Module):#utility class for training and testing.Needs to be present
                          #in all notebooks since it is inherited by models.

    def training_step(self,batch):
        text,aspect,labels=batch
        out=self(text,aspect)
        #labels=int(labels)
        labels=labels.to(torch.long)
        loss=F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch):
        text,aspect,labels=batch
        out=self(text,aspect)
        labels=labels.to(torch.long)
        loss=F.cross_entropy(out,labels)
        acc=accuracy(labels,out)
        return {'val_loss':loss.detach(),'val_acc':acc}
    
    def validation_epoch_end(self,result):
        loss=[x['val_loss'] for x in result]
        acc= [x['val_acc'] for x in result]
        l_b=torch.stack(loss).mean()
        a_b=torch.stack(acc).mean()
        return {'val_loss':l_b.item(), 'val_acc': a_b.item()}
    
    def epoch_end(self,epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch,result['val_loss'], result['val_acc']))


# In[ ]:


class LSTM_imp(aspectClassificationBase):#BI-LSTM class
    def __init__(self,emb_mat,num_classes,embedding_dim,vocab_size_t,vocab_size_a):
        super(LSTM_imp,self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size_t = vocab_size_t
        self.vocab_size_a=vocab_size_a
        self.polarities=num_classes
        self.embed = nn.Embedding.from_pretrained(emb_mat,freeze=True)
        self.lstm =nn.LSTM(self.embedding_dim,128,batch_first=True,num_layers=2,dropout=0.2,bidirectional=True)
        #self.lstm_a  =nn.LSTM(self.embedding_dim,128,batch_first=True,bidirectional=True)
        self.dense = nn.Linear(self.vocab_size_t*self.vocab_size_a,self.polarities)
        #self.softmax=nn.Softmax(dim=1)

    def forward(self, text,aspect):
        #asp_raw_indices=inputs[:,278:,:]
       # y1=inputs[0][278:][:]
        #y2=inputs[1][278:][:]
        
        #text_raw_indices = torch.stack([x1,x2])
        #asp_raw_indices=torch.stack([y1,y2])
        #print(asp_raw_indices.shape)
        #print(text_raw_indices.shape)
        x= self.embed(text)
        x,_=self.lstm(x)
        #print(x.shape)
        
        x2=self.embed(aspect)
        x2,_=self.lstm(x2)
        #print(x2.shape)
        x2=x2.permute(0,2,1)
        
        x3=torch.matmul(x,x2)
        #print(x3.shape)
        x3=x3.reshape(batch_size,-1)
        x3=self.dense(x3)
        #print(x3.shape)
       
        #x_len = torch.sum(text_raw_indices != 0, dim=-1)
        #_, (h_n, _) = self.lstm(x, x_len)
        #out = self.dense(h_n[0])
        return x3


# In[ ]:


class LSTM_AOA(aspectClassificationBase):#AOA class
    def __init__(self,emb_mat,num_classes,embedding_dim,vocab_size_t,vocab_size_a):
        super(LSTM_AOA,self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size_t = vocab_size_t
        self.vocab_size_a=vocab_size_a
        self.polarities=num_classes
        self.embed = nn.Embedding.from_pretrained(emb_mat,freeze=True)
        self.lstm =nn.LSTM(self.embedding_dim,128,batch_first=True,num_layers=1,bidirectional=True)
        #self.lstm_a  =nn.LSTM(self.embedding_dim,128,batch_first=True,bidirectional=True)
        self.dense = nn.Linear(256,self.polarities)
        self.softmax_c=nn.Softmax(dim=2)
        self.softmax_r=nn.Softmax(dim=1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, text,aspect,batch_size):
        #asp_raw_indices=inputs[:,278:,:]
       # y1=inputs[0][278:][:]
        #y2=inputs[1][278:][:]
        
        #text_raw_indices = torch.stack([x1,x2])
        #asp_raw_indices=torch.stack([y1,y2])
        #print(asp_raw_indices.shape)
        #print(text_raw_indices.shape)
        x= self.embed(text)
        x,_=self.lstm(x)
       # print(x.shape)
        
        x2=self.embed(aspect)
        x2,_=self.lstm(x2)
        #print(x2.shape)
        x2=x2.permute(0,2,1)
        
        x3=torch.matmul(x,x2)
        #print(x3.shape)
        x4=x3.detach().clone()
        x5=x3.detach().clone()
        x5=self.softmax_c(x5)
        #print(x5.shape)
        x4=self.softmax_r(x4)
        #x4=x4.permute(0,2,1)
        x5=x5.sum(dim=1)/self.vocab_size_t
        #print(x5.shape)
        #print(x5)
        x5=x5.reshape(batch_size,self.vocab_size_a,1)
        x6=torch.matmul(x4,x5)
     
       # print(x6.shape)
       # print(x6.shape)
        x=x.permute(0,2,1)
       # print(x.shape)
        x7=torch.matmul(x,x6)
        #print(x7.shape)
        
        
        x7=x7.reshape(batch_size,-1)
        #print(x7.shape)
        x7=self.dense(x7)
        #print(x7.shape)
        #out=self.softmax(x7)
        #x_len = torch.sum(text_raw_indices != 0, dim=-1)
        #_, (h_n, _) = self.lstm(x, x_len)
        #out = self.dense(h_n[0])
        return x7


# In[ ]:


device=get_default_device()


# In[ ]:


#model class


# In[ ]:


test_dl=DeviceDataLoader(test_dl,device)#transferdataloader to device 


# **Load entire model**

# In[ ]:


model = torch.load('../input/d/enkrish259/aoa-colwise-softmax/aoa_b20_e30_entire_model.pth')#load entire model
to_device(model,device)
model.eval()


# > ****Load only model state_dict****

# In[ ]:


#model = model_name
#model.load_state_dict(torch.load(PATH))
#to_device(model,device)
#model.eval()


# In[ ]:


def predict(test_dl,model):
    ans=[]
    with torch.no_grad():
        for batch in test_dl:
            text,aspect=batch
            out=model(text,aspect,batch_size) #use this out for AOA LSTM
            #out=model(text,aspect) -use this one for LSTM_IMP
            #print(out)
        #print(out)
        #print(type(out))
            _, preds = torch.max(out, dim=1)
            #print(preds)
            preds=preds.cpu().detach().numpy()
            preds=preds.tolist()
            ans.extend(preds)
            
    label=pd.Series(ans)
    return ans  


# In[ ]:


#ans=predict(test_dl,model)


# In[ ]:


#len(ans)


# In[ ]:


data['label']=predict(test_dl,model)
data.drop(['text_tok','aspect_tok'],axis=1,inplace=True)
data.to_csv('lstm_aoa_col_row_softmax_test.csv')#saving to test file


# In[ ]:


data


# In[ ]:




