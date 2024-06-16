import torch as tc
import torch.nn as nn
import torch.nn.functional as functional
from layers import Encoder_layer,Decoder_layer
import numpy as np 

def get_mask_pad(inp_seq:tc.FloatTensor,pad_chr:int):
    return (inp_seq!=pad_chr).unsqueeze(-2)

def look_ahead_mask(inp_seq:tc.FloatTensor):
    N,seq_len=inp_seq.shape
    mask=(1-tc.triu(tc.ones(1,seq_len,seq_len),diagonal=1)).bool 
    return mask

class pos_enc(nn.Module):

    def __init__(self,embd_dim,positions=150):
        
        self.register_buffer("pos_enc",get_pos_enc(embd_dim,positions))

        def get_pos_enc(dim,pos):
            
            def get_angles(position,dim,):
                return [position/(10000**(2*(i//2)/dim)) for i in range(dim)]
            
            position_angles=np.array([get_angles(position) for position in pos])
            position_angles[:,0::2]=np.sin(position_angles[:,0::2])
            position_angles[:,1::2]=np.cos(position_angles[:,1::2])

            return tc.FloatTensor(position_angles).unsqueeze(0)
    
    def forward(self,inp_seq):
        inp_seq+=pos_enc[:,:inp_seq.shape[1]].clone().detach()
        return inp_seq

class Encoder(nn.Module):

    def __init__(self,src_vocab_len,embd_dim,pad_idx,n_layers,n_heads,dk,dv,hidden_dim,dropout=0.1,seq_len=200,scale_emb=False):

        super().__init__()

        self.inp_embd=nn.Embedding(src_vocab_len,embd_dim,padding_idx=pad_idx)
        self.positional_enc=pos_enc(embd_dim,positions=seq_len)
        
        self.layer_stack=nn.ModuleList([Encoder_layer(embd_dim,heads=n_heads,d_k=dk,d_v=dv,hidden_dim=hidden_dim) 
                                        for s in range(n_layers)])
        
        self.layernorm=nn.LayerNorm(embd_dim,eps=1e-9)
        self.dropout=nn.Dropout(dropout)
        self.scale_emb=scale_emb
        self.embd_dim=embd_dim

        
    def forward(self,inp_seq,inp_mask,return_attns=False):

        attns=[]
        src_embd=self.inp_embd(inp_seq)
        if self.scale_emb:
            src_embd*=self.embd_dim**0.5
        src_with_pos=self.positional_enc(src_embd)
        src_inp=self.dropout(src_with_pos)
        enc_out=self.layernorm(src_inp)

        for layer in self.layer_stack:
            enc_out,self_attn=layer(enc_out,inp_mask)
            if return_attns:
                attns.append(self_attn)
        
        if return_attns:
            return enc_out,self_attn
        
        return enc_out
    
class Decoder(nn.Module):

    def __init__(self,src_vocab,embd_dim,pad_idx,n_heads,n_layers,dk,dv,hidden_dim,dropout=0.1,seq_len=200,scale_emb=False):
        super().__init__()

        self.inp_embd=nn.Embedding(src_vocab,embd_dim,padding_idx=pad_idx)
        self.pos_enc=pos_enc(embd_dim,seq_len)
        
        self.layer_stack=nn.ModuleList([Decoder_layer(embd_dim,n_heads,dk,dv,hidden_dim) 
                                        for t in range(n_layers)])
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(embd_dim,eps=1e-9)
        self.embd_dim=embd_dim
        self.scale_emb=scale_emb

    def forward(self,dec_input,enc_output,src_attn_mask=None,dec_attn_mask=None,return_attns=False):

        dec_self_attn,dec_enc_attn=[],[]
        dec_inp=self.inp_embd(dec_input)
        dec_inp=self.pos_enc(dec_inp)

        if self.scale_emb:
            dec_inp*=self.embd_dim**0.5
        dec_drop=self.dropout(dec_inp)
        dec_out=self.layernorm(dec_drop)

        for layer in self.layer_stack:
            dec_out,dec_self_a,dec_enc_a=layer(dec_out,enc_output,self_attn_mask=src_attn_mask,dec_attn_mask=dec_attn_mask)
            if return_attns:
                dec_self_attn.append(dec_self_a)
                dec_enc_attn.append(dec_enc_a)

        if return_attns:
            return dec_out,dec_self_attn,dec_enc_attn
        return dec_out
    
    


            



