''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from lib.model.layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
import lib.utils.utils as utils

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x,n_person):
        p=self.pos_table[:,:x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        p=self.pos_table2[:, :int(x.shape[1]/n_person)].clone().detach()
        p=p.repeat(1,n_person,1)
        return x + p


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, device='cuda'):

        super().__init__()
        self.position_embeddings = nn.Embedding(n_position, d_model)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device
        
        
    def forward(self, src_seq,n_person, src_mask, return_attns=False, global_feature=False):
        enc_slf_attn_list = []
        if global_feature:
            enc_output = self.dropout(self.position_enc.forward2(src_seq,n_person))
        else:
            enc_output = self.dropout(self.position_enc(src_seq,n_person))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1,device='cuda'):

        super().__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device=device

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100,
            k_levels=1, share_d=False, device='cuda'):

        super().__init__()
        
        self.device=device
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj=nn.Linear(45,d_model) # 45: 15jointsx3
        self.proj2=nn.Linear(45,d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.share_d = share_d

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, device=self.device)

        self.k_levels = k_levels

        if share_d:
            depth = 2

        else:
            depth = k_levels + 1

        self.decoders = nn.ModuleList([
            Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, device=self.device)
            for i in range(depth)])

        self.l1s = nn.ModuleList([
            nn.Linear(d_model, d_model*4)
            for i in range(depth)])

        self.l2s = nn.ModuleList([
            nn.Linear(d_model*4, d_model*25)
            for i in range(depth)])
        
        self.projs = nn.ModuleList([
            nn.Linear(d_model, 45)
            for i in range(depth)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
    
    def forward(self, src_seq, trg_seq, input_seq, use=None):
        '''
        src_seq: local
        trg_seq: local
        input_seq: global
        '''
        
        bs, n_person=input_seq.shape[:2]
        
        src_seq_=self.proj(src_seq)
        trg_seq_=self.proj2(trg_seq)

        enc_local, *_ = self.encoder(src_seq_, n_person, None)
        others=input_seq[:,:,:,:].view(bs,-1,45)
        others_=self.proj2(others)
        mask_other=None
        mask_dec=None
        results = []

        for i in range(self.k_levels+1):
            if self.share_d and i > 1:
                idx = -1
            else:
                idx = i

            decoder = self.decoders[idx]
            l1 = self.l1s[idx]
            l2 = self.l2s[idx]
            proj_inverse = self.projs[idx]
            
            if i==0:
                enc_global,*_=self.encoder_global(others_,n_person, mask_other, global_feature=True)  
                enc_global=enc_global.unsqueeze(1).expand(bs,input_seq.shape[1],-1,self.d_model)
                enc_global=enc_global.reshape(bs*n_person,-1,self.d_model)                         
            else:
                enc_global = dec_output.reshape(bs, n_person, -1, self.d_model)
                enc_global = enc_global.unsqueeze(1)
                enc_global = enc_global.expand(bs, n_person, n_person, -1, -1)
                enc_global = enc_global.reshape(bs*n_person,-1, self.d_model)

            enc_output = torch.cat([enc_local, enc_global],dim=1)
            
            dec_output, dec_attention, *_ = decoder(trg_seq_[:,:1,:], None, enc_output, mask_dec)   
            dec_output = l1(dec_output)
            dec_output = l2(dec_output)
            dec_output = dec_output.reshape(dec_output.shape[0],25,self.d_model)
            pred = proj_inverse(dec_output)
            results.append(pred.clone())
        return results

class Discriminator(nn.Module):
    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2, n_position=100, d_hidden=1024,
            device='cuda'):

        super().__init__()
        self.device=device     
        self.d_model=d_model
        self.encoder = Encoder(
                n_position=n_position,
                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                pad_idx=src_pad_idx, dropout=dropout, device=self.device)
        self.fc1 = nn.Linear(45, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x_in, n_person=5):
        x, *_ = self.encoder(x_in, n_person=n_person, src_mask=None, global_feature=True)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def calc_gen_loss(self, x_gen):
        fake_logits = self.forward(x_gen)
        loss = utils.disc_l2_loss(fake_logits)
        return loss
    
    def calc_dis_loss(self, x_gen, x_real):
        fake_logits = self.forward(x_gen)
        real_logits = self.forward(x_real)
        loss = utils.adv_disc_l2_loss(fake_logits, real_logits)
        return loss