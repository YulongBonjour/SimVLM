import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#helpers
def clone(module,N):
    '''copy the given module N times'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def subsequent_mask(size):
    attn_shape=(1,size,size)
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype(bool)
    return torch.from_numpy(subsequent_mask)==False

class Feedforward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(Feedforward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.w_2(self.dropout(F.relu((self.w_1(x)))))

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class Transformer(nn.Module):
    def __init__(self,d_model=512,heads=8,enc_depth=8,dec_depth=8,d_ff=1024,src_mask=None,tgt_mask=None,dropout=0.1):#,src_embeded,tgt_embed,generator):
        super(Transformer,self).__init__()
        c=copy.deepcopy
        attn=MultiHeadedAttention(heads,d_model)
        ff=Feedforward(d_model,d_ff,dropout)
        self.encoder=Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),enc_depth)
        self.decoder=Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),dec_depth)
        self.register_buffer('src_mask', src_mask, persistent=False)
        self.register_buffer('tgt_mask', tgt_mask, persistent=False)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self,src_embeded,tgt_embeded):#,src_mask,tgt_mask):
        return  self.decode(self.encode(src_embeded),tgt_embeded)
    def encode(self,src_embeded):#,src_mask):
        return self.encoder(src_embeded,self.src_mask)
    def decode(self,memory,tgt_embeded):
        return self.decoder(tgt_embeded,memory,self.src_mask,self.tgt_mask)

class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj=nn.Linear(d_model,vocab)
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
class Encoder(nn.Module):
    def __init__(self,layer,N):
        '''N encoder layers '''
        super(Encoder,self).__init__()
        self.layers=clone(layer,N)
        self.norm=LayerNorm(layer.size)
    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    '''LayerNorm +subLayer+dropout+residual connection'''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        '''size is the embedding dimension'''
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clone(SublayerConnection(size,dropout),2)
        self.size=size
    def forward(self,x,mask=None):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers=clone(layer,N)
        self.norm=LayerNorm(layer.size)
    def forward(self,x, memory,src_mask=None,tgt_mask=None):
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clone(SublayerConnection(size,dropout),3)
    def forward(self,x,memory,src_mask=None,tgt_mask=None):
        m=memory
        x=self.sublayer[0](x,lambda  x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x :self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)

def attention(query,key,value,mask=None,dropout=None):
     d_k=query.size(-1)
     scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
     if mask is not None:
         scores=scores.masked_fill(mask==0,-1e9)
     p_attn=F.softmax(scores,dim=-1)
     if dropout is not None:
         p_attn=dropout(p_attn)
     return torch.matmul(p_attn,value),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model//h
        self.h=h
        self.linears=clone(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
        '''print('q:',query)
        print('k:',key)
        print('v:',value)'''
        nbatchs=query.size(0)
        query, key, value=[l(x).view(nbatchs,-1,self.h,self.d_k).transpose(1,2)\
                           for l, x in zip(self.linears,(query,key,value))]
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous().view(nbatchs,-1,self.h*self.d_k)
        return self.linears[-1](x)

