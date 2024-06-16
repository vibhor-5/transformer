import torch.nn as nn
import torch.nn.functional as functional
from sublayers import SelfAttention,FeedForward

class Encoder_layer(nn.Module):
    def __init__(self,embd_dim,heads,d_k,d_v,hidden_dim):

        super().__init__()

        self.selfattention=SelfAttention(embd_dim,heads,d_k,d_v)
        self.FFN=FeedForward(embd_dim,hidden_dim)

    def forward(self,enc_input,mask=None): 
        enc_output,enc_self_attention=self.selfattention(enc_input,enc_input,enc_input,mask)
        enc_output=self.FFN(enc_output)
        
        return enc_output,enc_self_attention
    
class Decoder_layer(nn.Module):

    def __init__(self,embd_dim,n_heads,d_k,d_v,hidden_dim):

        super.__init__()

        self.self_attention=SelfAttention(embd_dim,n_heads,d_k,d_v)
        self.enc_attention=SelfAttention(embd_dim,n_heads,d_k,d_v)
        self.FFN=FeedForward(embd_dim,hidden_dim)

    def forward(self,dec_input,enc_output,self_attn_mask=None,dec_attn_mask=None):

        dec_output,dec_attention=self.self_attention(dec_input,dec_input,dec_input,self_attn_mask)
        dec_output,dec_enc_attn=self.enc_attention(dec_output,enc_output,enc_output,dec_attn_mask)
        dec_output=self.FFN(dec_output)

        return dec_output,dec_attention,dec_enc_attn
    
    

    


