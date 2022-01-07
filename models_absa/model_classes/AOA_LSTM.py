#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
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
pretrained_bert="bert-base-uncased"


# In[ ]:


class aspectClassificationBase(nn.Module):

    def training_step(self,batch):#batch wise training
        model.train()
        text,aspect,labels=batch
        out=self(text,aspect)
        #labels=int(labels)
        labels=labels.to(torch.long)
        loss=F.cross_entropy(out,labels)
        return loss
    
    def validation_step(self,batch):#batch wise validation
        model.eval()
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


class LSTM_AOA(aspectClassificationBase):#model class for Attention over attention
    def __init__(self,emb_mat,num_classes,embedding_dim,vocab_size_t,vocab_size_a):#emb_mat stores glove embeddings of all 
                                                                                   #words in our dictionary
        super(LSTM_AOA,self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size_t = vocab_size_t
        self.vocab_size_a=vocab_size_a
        self.polarities=num_classes
        self.embed = nn.Embedding.from_pretrained(emb_mat,freeze=True)#use glove word embedding matrix 
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
        x7=self.dense(x7)#no need for softmax if cross entropy loss is being used as a loss function.
        #print(x7.shape)
        #out=self.softmax(x7)
        #x_len = torch.sum(text_raw_indices != 0, dim=-1)
        #_, (h_n, _) = self.lstm(x, x_len)
        #out = self.dense(h_n[0])
        return x7

