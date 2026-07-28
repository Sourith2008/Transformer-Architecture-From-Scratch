"""Importing Modules"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
print("Modules imported")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Scaled Dot Product"""

def scaled_dot_product(q,k,v,mask=None):
  d_k=q.size()[-1]
  scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
  if mask is not None:
    scaled+=mask
  attention=F.softmax(scaled,dim=-1)
  values=torch.matmul(attention,v)
  return attention,values
print("Scaled dot product defined successfully")

"""Multi Head Attention"""

class MultiHeadAttention(torch.nn.Module):
  def __init__(self,num_heads,d_model,max_len):
    super().__init__()
    self.num_heads=num_heads
    self.d_model=d_model
    self.max_len=max_len
    self.head_dim=d_model//num_heads
    self.qkv_layer=nn.Linear(d_model,3*d_model)
    self.linear=nn.Linear(d_model,d_model)
  def forward(self,x):
    batch_size,seq_len,d_model=x.size()
    qkv=qkv_layer(x)
    qkv==qkv.reshape(batch_size,seq_len,num_heads,3*self.head_dim)
    qkv=qkv.permute(0,2,1,3)
    q,k,v=qkv.chunk(3,dim=-1)
    attention,values=scaled_dot_product(q,k,v)
    values=values.permute(0,2,1,3).reshape(batch_size,seq_len,self.num_heads*self.head_dim)
    out=self.linear(values)
    return out
print("Multihead attention defined successfully")

"""Positional Encoding"""

class Positional_Encoding(nn.Module):
    def __init__(self,d_model,max_len):
        super().__init__()
        self.d_model=d_model
        self.max_len=max_len
    def forward(self):
        i=torch.arange(0,self.d_model,2)
        pos=torch.arange(self.max_len)
        denom=10000**(i/self.d_model)
        sin=torch.sin(pos/denom)
        cos=torch.cos(pos/denom)
        stacked=torch.stack((sin,cos),dim=2)
        PE=torch.flatten(stacked,start_dim=1,end_dim=2)
        return PE
    print("Positional encoding defined successfully")

"""Layer Normalization"""

class LayerNormalization(nn.Module):
  def __init__(self,param_shape,eps=1e-5):
    self.param_shape=param_shape
    self.eps=eps
    self.gamma=nn.Parameter(torch.ones(param_shape))
    self.beta=nn.Parameter(torch.zeros(param_shape))
  def forwars(self,inputs):
    dims=[-(i+1) for i in range(len(self.param_shape))]
    mean=inputs.mean(dim=dims,keepdim=True)
    var=((inputs-mean)**2).mean(dim=dims,keepdim=True)
    std=(var+self.eps).sqrt()
    y=(inputs-mean)/std
    out=self.gamma*y+self.beta
    return out
print("LayerNormalization defined successfully!")

"""Feed Forward Network"""

class FeedForward(nn.Module):
  def __init__(self,d_model,ffn_hidden):
    self.d_model=d_model
    self.ffn_hidden=ffn_hidden
    self.linear1=nn.Linear(d_model,ffn_hidden)
    self.linear2=nn.Linear(ffn_hidden,d_model)
    self.dropout1=nn.Dropout(p=0.1)
    self.dropout2=nn.Dropout(p=0.1)
  def forward(self,x):
    x=self.linear1(x)
    x=F.relu(x)
    x=self.dropout1(x)
    x=self.linear2(x)
  print("Feed Forward Network defined successfully!")

"""Multi Head Cross Attention"""

class MultiHeadCrossAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.head_dim=d_model//num_heads
    self.q_layer=nn.Linear(d_model,d_model)
    self.kv_layer=nn.Linear(d_model,d_model*2)
    self.linear=nn.Linear(d_model,d_model)
  def forward(self,x,y):
    batch_size,seq_len,d_model=x.size()
    q=self.q_layer(y)
    kv=self.kv_layer(x)
    kv=kv.resahpe(batch_size,seq_len,self.num_heads,2*self.head_dim)
    q=q.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
    q=q.permute(0,2,1,3)
    kv=kv.permute(0,2,1,3)
    k,v=kv.chunk(2,dim=-1)
    attention,value=scaled_dot_product(q,k,v)
    value=value.permute(0,2,1,3).reshape(batch_size,seq_len,d_model)
    out=self.linear(value)
    return out
print("Multi Head Cross Attention defined successfully!")

"""Sentence Embedding"""

class SentenceEmbedding(nn.Module):
  def __init__(self,d_model,max_len,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
    super().__init__()
    self.d_model=d_model
    self.max_len=max_len
    self.embedding=nn.Embedding(max_len,d_model)
    self.pe=Positional_Encoding(d_model=d_model,max_len=max_len)
    self.language_to_index=language_to_index
    self.START_TOKEN=START_TOKEN
    self.END_TOKEN=END_TOKEN
    self.PADDING_TOKEN=PADDING_TOKEN
    self.dropout=nn.Dropout(p=0.1)
  def batch_tokenize(self,batch,start_token,end_token):
    def tokenize(sentence,start_token,end_token):
      sentence_word_indices=[self.language_to_index[token] for token in list(sentence)]
      if start_token:
        sentence_word_indices.insert(0,self.language_to_index[self.START_TOKEN])
      if end_token:
        sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
      for  _ in range(len(sentence_word_indices),self.max_len):
        sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
      return torch.tensor(sentence_word_indices)
    tokenized=[]
    for sentence_num in range(len(batch)):
      tokenized.append(tokenize(batch[sentence_num],start_token,end_token))
    tokenized=torch.stack(tokenized)
    return tokenized.to(device)
  def forward(self,x,start_token,end_token):
    x=batch_tokenize(x,start_token,end_token)
    x=self.embedding(x)
    pos=self.pe().to(device)
    x=x+pos
    x=self.dropout(x)
    return x
print("Senetence Embedding defined successfully!")

"""ENCODER"""

class EncoderLayer(nn.Module):
  def __init__(self,d_model,ffn_hidden,num_heads,drop_prob=0.1):
    super(EncoderLLayer,self).__init__()
    self.d_model=d_model
    self.ffn_hidden=ffn_hidden
    self.drop_prob=drop_prob
    self.num_heads=num_heads
    self.attention=MultiHeadAttention(num_heads=num_heads,d_model=d_model)
    self.layer_norm1=LayerNormalization(param_shape=[d_model])
    self.layer_norm2=LayerNormalization(param_shape=[d_model])
    self.dropout=nn.Dropout(p=drop_prob)
    self.ffn=FeedForward(d_model=d_model,ffn_hidden=ffn_hidden)
  def forward(self,x):
    _x=x
    x=self.attention(x)
    x=self.dropout(x)
    x=self.layer_norm1(x+_x)
    _x=x
    x=self.ffn(x)
    x=self.dropout(x)
    x=self.layer_norm2(x+_x)
    return x
print("Encoder Layer defined successfully!")

class SequentialEncoder(nn.Sequential):
  def forward(self,*inputs):
    x,self_attention_mask=inputs
    for module in self._modules.values():
      x=module(x,self_attention_mask)
    return x
print("Sequential Encoder defined successfully!")

class Encoder(nn.Module):
  def __init__(self,d_model,max_len,language_to_index,ffn_hidden,num_heads,num_layers,START_TOKEN,END_TOKEN,PADDING_TOKEN):
    super().__init__()
    self.d_model=d_model
    self.max_len=max_len
    self.ffn_hidden=ffn_hidden
    self.num_layers=num_layers
    self.num_heads=num_heads
    self.START_TOKEN=START_TOKEN
    self.END_TOKEN=END_TOKEN
    self.PADDING_TOKEN=PADDING_TOKEN
    self.sentence_embedding=SentenceEmbedding(d_model=d_model,max_len=max_len,language_to_index=language_to_index,START_TOKEN=START_TOKEN,END_TOKEN=END_TOKEN,PADDING_TOKEN=PADDING_TOKEN)
    self.layers=SequentialEncoder(*[EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,num_heads=num_heads,drop_prob=drop_prob) for _ in range(num_layers)])
  def forward(self,x,self_attention_mask):
    x=self.sentence_embedding(x)
    x=self.layers(x)
    return x
print("Encoder defined successfully!")

"""DECODER"""

class DecoderLayer(nn.Module):
  def __init__(self,d_model,num_heads,max_len,ffn_hidden):
    super().__init__()
    self.d_model=d_model
    self.max_len=max_len
    self.num_heads=num_heads
    self.attention=MultiHeadAttention(num_heads=num_heads,d_model=d_model)
    self.dropout1=nn.Dropout(p=0.1)
    self.dropout2=nn.Dropout(p=0.1)
    self.dropout3=nn.Dropout(p=0.1)
    self.cross_attention=MultiHeadCrossAttention(d_model=d_model,num_heads=num_heads)
    self.norm1=LayerNormalization(param_shape=[d_model])
    self.norm2=LayerNormalization(param_shape=[d_model])
    self.norm3=LayerNormalization(param_shape=[d_model])
    self.ffn=FeedForward(d_model=d_model,ffn_hidden=ffn_hidden)
  def Forward(self,x,y):
    _y=y
    y=self.attention(y,self_attention_mask,cross_attention_mask)
    y=self.dropout1(y)
    y=self.norm1(y+_y)
    _y=y
    y=self.cross_attention(x,y,mask=cross_attention_mask)
    y=self.dropout2(y)
    y=self.norm2(y+_y)
    _y=y
    y=self.ffn(y)
    y=self.dropout3(y)
    y=self.norm3(y+_y)
    return y
print("Decoder Layer defined successfully")

class SequentialDecoder(nn.Sequential):
  def forward(self,*inputs):
    x,y,self_attention_mask,cross_attention_mask=inputs
    for module in self._modules.values():
      y=module(x,y,self_attention_mask,cross_attention_mask)
    return y
print("Sequential Decoder defined successfully!")

class Decoder(nn.Module):
  def __init__(self,d_model,max_len,ffn_hidden,num_heads,num_layers,language_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
    super().__init__()
    self.embedding=SentenceEmbedding(d_model=d_model,language_to_index=language_to_index,max_len=max_len,START_TOKEN=START_TOKEN,END_TOKEN=END_TOKEN,PADDING_TOKEN=PADDING_TOKEN)
    self.layers=SequentialDecoder(*[DecoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,num_heads=num_heads,max_len=max_len) for _ in range(num_layers)])
  def forward(self,x,y,self_attention_mask,cross_attention_mask,start_token,end_token):
    y=self.embedding(y,start_token,end_token)
    y=self.layers(x,y,self_attention_mask,cross_attention_mask)
    return y
print("Decoder defined successfully!")

"""THE TRANSFORMER CLASS"""

class Transformer(nn.Module):
  def __init__(self,d_model,max_len,num_heads,ffn_hidden,num_layers,bn_vocab_size,bengali_to_index,english_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
    super().__init__()
    self.encoder=Encoder(d_model,max_len,english_to_index,ffn_hidden,num_heads,num_layers,START_TOKEN,END_TOKEN,PADDING_TOKEN)
    self.decoder=Decoder(d_model,max_len,ffn_hidden,num_heads,num_layers,bengali_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
    self.linear=nn.Linear(d_model,bn_vocab_size)
  def forward(self,x,y,encoder_self_attention_mask=None,decoder_self_attention_mask=None,decoder_cross_attention_mask=None):
    x=self.encoder(x,encoder_self_attention_mask)
    y=self.decoder(x,y,decoder_self_attention_mask,decoder_cross_attention_mask)
    y=self.linear(y)
    return y
print("Transformer defined successfully!")