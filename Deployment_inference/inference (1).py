import json
import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
#from nltk.tokenize import WordPunctTokenizer
import pickle

logger = logging.getLogger(__name__)

class LSTM_imp(nn.Module):
    def __init__(self,num_classes,embedding_dim,vocab_size_t,vocab_size_a):
        super(LSTM_imp,self).__init__()
        self.embedding_dim = embedding_dim
        #self.hidden_dim = hidden_dim
        self.vocab_size_t = vocab_size_t
        self.vocab_size_a=vocab_size_a
        self.polarities=num_classes
        
        self.lstm_t  =nn.LSTM(self.embedding_dim,128,batch_first=True,bidirectional=True)
        self.lstm_a  =nn.LSTM(self.embedding_dim,128,batch_first=True,bidirectional=True)
        self.dense = nn.Linear(self.vocab_size_t*self.vocab_size_a,self.polarities)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, inputs):
        text_raw_indices=inputs[:,:278,:]
        asp_raw_indices=inputs[:,278:,:]

        x,_=self.lstm_t(text_raw_indices)
        
        x2,_=self.lstm_a(asp_raw_indices)
        x2=x2.permute(0,2,1)
        
        x3=torch.matmul(x,x2)
        x3=x3.reshape(2,-1)
        x3=self.dense(x3)
        out=self.softmax(x3)
        
        return out

def get_vocab(bucket, key):

    path = f's3://{bucket}/{key}'
    with open(path, 'rb') as handle:
        vocab = pickle.load(handle)
    return vocab 

def get_inputs(data, bucket, key):
    w_ind = get_vocab(bucket, key)
    sent, asp = data['context'], data['aspect']

    encoded_sentences = np.array([w_ind[word] for word in sent])
    encoded_sen=np.append(encoded_sentences,np.zeros(278-len(encoded_sentences)))
    encoded_asp=np.array([w_ind[word] for word in asp])
    encoded_as=np.append(encoded_asp,np.zeros(8-len(encoded_asp)))
    return encoded_sen, encoded_as

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model....')
    model  =LSTM_imp(num_classes=3,embedding_dim=300,vocab_size_t=278,vocab_size_a=8)

    #https://sagemaker-absa.s3.us-east-2.amazonaws.com/MODEL.pth
    path = 's3:\\sagemaker-absa\MODEL.pth'
    #with open(path, 'rb') as f:
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model

def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        aspect, context = input_data['aspect'], input_data['context']
        logger.info(f'Aspect is: {aspect}','\n', f"Context is: {context}")
        
        #https://sagemaker-absa.s3.us-east-2.amazonaws.com/VOCABx.pickle
        return get_inputs({'aspect':aspect, 'context':context},'sagemaker-absa','VOCABx.pickle')
    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.....')
    result  = torch.argmax(prediction_output[0])

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')


def predict_fn(input_data, model):
    

    return model(input_data[0], input_data[1])