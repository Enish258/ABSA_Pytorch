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


class AspectC(nn.Module):#utility class for training and testing.Needs to be present
                          #in all notebooks since it is inherited by models.

    def training_step(self,batch):
        model.train()
        concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices,labels=batch

        out=self(concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices)
        #labels=int(labels)
        labels=labels.to(torch.long)
        loss=criterion(out,labels)
        #print('train')
        #print(loss)
        return loss
    
    def validation_step(self,batch):
        model.eval()
        concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices,labels=batch
    #print(batch)
    #print(text)
    #out=bert(text_bert_indices)
        out=self(concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices)
        labels=labels.to(torch.long)
        loss=criterion(out,labels)
        acc=accuracy(labels,out)
        #print('val')
        #print(loss)
        
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


class SelfAttention(AspectC):
    def __init__(self, config, device,max_seq_len):
        super(SelfAttention,self).__init__()
        #self.opt = opt
        self.max_seq_len=max_seq_len
        self.device=device
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(np.zeros((inputs.size(0), 1, 1, self.max_seq_len),
                                            dtype=np.float32), dtype=torch.float32).to(self.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class CUST_BERT(AspectC):
    def __init__(self, bert,dropout,bert_dim,device,max_seq_len,lcf):
        super(CUST_BERT, self).__init__()

        self.bert_spc = bert
        self.max_seq_len=max_seq_len
       # self.opt = opt
        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = bert
        self.local_context_focus=lcf
        self.SRD=3
        self.device=device
        self.bert_dim=bert_dim# Default to use single Bert and reduce memory requirements
        self.dropout = nn.Dropout(dropout)
        self.bert_SA = SelfAttention(bert.config, device,self.max_seq_len)
        self.linear_double = nn.Linear(self.bert_dim * 2, self.bert_dim)
        self.linear_single = nn.Linear(self.bert_dim, self.bert_dim)
       # self.BatchNorm=nn.BatchNorm1d(self.max_seq_len)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(self.bert_dim,3)
        #self.softmax=nn.Softmax(dim=1)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.max_seq_len, self.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros((self.bert_dim), dtype=np.float)
            for j in range(asp_begin + asp_len + mask_len, self.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros((self.bert_dim), dtype=np.float)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.max_seq_len, self.bert_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.SRD:
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2
                                        - self.SRD)/np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.device)

    def forward(self,concat_bert_indices,concat_segments_indices,text_loc_indices,aspect_bert_indices):
        text_bert_indices = concat_bert_indices
        bert_segments_ids = concat_segments_indices
        text_local_indices = text_loc_indices
        aspect_indices = aspect_bert_indices

        bert_spc_out= self.bert_spc(text_bert_indices, token_type_ids=bert_segments_ids)
        #print(bert_spc_out)
        bert_spc_out = self.dropout(bert_spc_out[0])

        bert_local_out= self.bert_local(text_local_indices)
        #print(bert_local_out)
        #print(bert_spc_out)
        bert_local_out = self.dropout(bert_local_out[0])

        if self.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices)
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        #mean_pool=self.BatchNorm(mean_pool)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        #dense_out=self.softmax(dense_out)

        return dense_out


# In[ ]:





# In[ ]:




