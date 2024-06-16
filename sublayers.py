import torch
import torch.nn as nn
import torch.nn.functional as functional

class SelfAttention(nn.Module):
    
    def __init__(self,embd_dim,heads,d_k,d_v,dropout_rate1=0.2,dropout_rate2=0.2):

        super().__init__()

        self.emd_dim=embd_dim
        self.heads=heads
        self.d_k=d_k
        self.d_v=d_v
        self.head_dim=embd_dim//heads
        assert(self.head_dim*self.heads==self.embd_dim),"no. of heads should be divisble by embd_dim"

        self.w_q=nn.Linear(embd_dim,self.d_k*self.heads,bias=False)
        self.w_k=nn.Linear(embd_dim,self.d_k*self.heads,bias=False)
        self.w_v=nn.Linear(embd_dim,self.d_v*self.heads,bias=False)
        self.dropout1=nn.Dropout(dropout_rate1)
        self.dropout2=nn.Dropout(dropout_rate2)
        self.norm=nn.LayerNorm(embd_dim,eps=1e-6)
    def forward(self,query,key,value,mask=None):

        N=query.shape[0]
        heads=self.heads
        d_k=self.d_k
        d_v=self.d_v

        query_len=query.shape[1] #query=(N,q_len,embd_dim)
        key_len=key.shape[1]
        value_len=value.shape[1]
        res=query

        Q=self.w_q(query).view(N,query_len,heads,d_k).transpose(1,2) #size=(N,heads,q_len,head_dim)
        K=self.w_k(key).view(N,key_len,heads,d_k).transpose(1,2) #size=(N,heads,k_len,head_dim)
        V=self.w_v(value).view(N,value_len,heads,d_v).transpose(1,2) #size=(N,heads,v_len,head_dim)
        
        
        E=functional.softmax((Q@K.mT)/(d_k**0.5),dim=-1)

        if mask is not None:
            mask=mask.unsqueeze(1)
            E.masked_fill_(mask==0,value=-1e9)
        E_d=self.dropout(E)
        attention= E_d@V #shape=(N,heads,q_len,head_dim)
        attention= attention.transpose(1,2).contiguous().view(N,query_len,-1)
        attention+=res
        attention=self.norm(attention)
        return attention,E_d
    

class FeedForward(nn.Module):
    
    def __init__(self,embd_dim,hidden_dim,dropout_rate=0.1):
        super().__init__()

        self.in_dim=embd_dim
        self.hidden_dim=hidden_dim
        self.linear1=nn.Linear(self.in_dim,self.hidden_dim)
        self.Linear2=nn.Linear(self.hidden_dim,self.in_dim)
        self.dropout=nn.Dropout(dropout_rate)
        self.norm=nn.LayerNorm(self.in_dim,eps=1e-6)

    def forward(self,attn):
        res=attn
        attn=self.linear2(functional.relu(self.linear1(attn)))
        attn=self.dropout(attn)
        attn+=res
        attn=self.norm(attn)
        return attn
    

