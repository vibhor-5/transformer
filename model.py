import torch as tc
import torch.nn as nn 
from blocks import get_mask_pad,look_ahead_mask,Encoder,Decoder

class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,src_voc_len,trg_voc_len,n_layers=6,embd_dim=512,n_position=200,
                 n_heads=8,dk=64,dv=64,hidden_dim=1024,dropout=0.1, scale_or_prj="scale",share_emb_wt=True,share_emb_dense_wt=True):

        super().__init()

        self.src_pad_idx,self.trg_pad_idx=src_pad_idx,trg_pad_idx

        assert scale_or_prj in ['scale','project','None'],"scale,project or none for scale_or_proj"
        self.scale_emb=(scale_or_prj=="scale") if share_emb_wt else False
        self.prj_emb=(scale_or_prj=="project") if share_emb_wt else False
        self.embd_dim=embd_dim

        self.encoder=Encoder(src_voc_len,embd_dim,self.src_pad_idx,n_layers,n_heads,dk,dv,
                             hidden_dim,dropout,seq_len=n_position,scale_emb=self.scale_emb)
        
        self.decoder=Decoder(trg_voc_len,embd_dim=self.embd_dim,pad_idx=self.trg_pad_idx,n_heads=n_heads,n_layers=n_layers,
                             dk=dk,dv=dv,hidden_dim=hidden_dim,dropout=dropout,seq_len=n_position,scale_emb=self.scale_emb)
        
        self.final=nn.Linear(self.embd_dim,trg_voc_len,bias=False)

        for p in self.parameters:
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

        if share_emb_wt:
            self.encoder.inp_embd.weight=self.decoder.inp_embd.weight
        if share_emb_dense_wt:
            self.final.weight=self.decoder.inp_embd.weight

    def forward(self,inp_seq,out_seq):
        src_mask=get_mask_pad(inp_seq=inp_seq,pad_chr=self.src_pad_idx)
        trg_mask= get_mask_pad(out_seq,self.trg_pad_idx) & look_ahead_mask(out_seq)
        enc_out,*_= self.encoder(inp_seq,src_mask)
        dec_out,*_=self.decoder(out_seq,enc_out,src_attn_mask=trg_mask,dec_attn_mask=src_mask)
        trg_logit=self.final(dec_out)

        if self.prj_emb:
            trg_logit/=self.embd_dim**0.5
        return trg_logit
    




