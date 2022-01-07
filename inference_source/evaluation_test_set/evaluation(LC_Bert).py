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


# Import libraries

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


    
test_2=test.copy()
data=parse_data(test_2)

#print(data)
max_tex,max_asp=max_len(data)
max_tex=278
max_asp=8


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


device=get_default_device()


# In[ ]:


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer_Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


# In[ ]:


tokenizer=Tokenizer_Bert(max_tex,pretrained_bert)


# In[ ]:


data


# In[ ]:


def dataset_creator(text,aspect,text_len,aspect_len):
    concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text+ ' [SEP] ' + aspect + " [SEP] ")
    concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
    concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

    text_bert_indices = tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")
    return concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices


# In[ ]:


len(data)


# In[ ]:


def final_dataset(data):
    concat_bert_indices=np.empty([len(data),max_tex])
    concat_segments_indices=np.empty([len(data),max_tex])
    text_bert_indices=np.empty([len(data),max_tex])
    aspect_bert_indices=np.empty([len(data),max_tex])
    for i in range(len(data)):#creating the data to the form required by models
        sent=data.loc[i,'text']
        asp=data.loc[i,'aspect']
        
        sent_len=len(data.loc[i,'text_tok'])
        asp_len=len(data.loc[i,'aspect_tok'])
        concat_bert_indices_1,concat_segments_indices_1,text_bert_indices_1,aspect_bert_indices_1=dataset_creator(sent,asp,sent_len,asp_len)
        concat_bert_indices[i]=concat_bert_indices_1
        concat_segments_indices[i]=concat_segments_indices_1
        text_bert_indices[i]=text_bert_indices_1
        aspect_bert_indices[i]=aspect_bert_indices_1
        
    return concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices#model requires 4 inputs


# In[ ]:


concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices=final_dataset(data)


# In[ ]:


concat_bert_indices=torch.LongTensor(concat_bert_indices)
concat_segments_indices=torch.LongTensor(concat_segments_indices)
text_bert_indices=torch.LongTensor(text_bert_indices)
aspect_bert_indices=torch.LongTensor(aspect_bert_indices)


# In[ ]:





# In[ ]:



#create dataset and dataloader from data
test_ds=TensorDataset(concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices)
batch_size = 8
test_dl = DataLoader(test_ds, batch_size, shuffle=False)


# In[ ]:


test_dl=DeviceDataLoader(test_dl,device)


# In[ ]:





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


model = torch.load([model_file])
to_device(model,device)#transfer model to device
model.eval()


# In[ ]:


def predict(test_dl,model):
    ans=[]
    with torch.no_grad():
        for batch in test_dl:
            concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices=batch
            out=model(concat_bert_indices,concat_segments_indices,text_bert_indices,aspect_bert_indices)
        #print(out)
        #print(type(out))
            _, preds = torch.max(out, dim=1)
            preds=preds.cpu().detach().numpy()
            preds=preds.tolist()
            ans.extend(preds)
            
    label=pd.Series(ans)
    return ans  #print(preds)


# In[ ]:


data['label']=predict(test_dl,model)
data.drop(['text_tok','aspect_tok'],axis=1,inplace=True)
data.to_csv([path])


# In[ ]:


data


# In[ ]:





# In[ ]:




