import torch
from torch import nn
import torch.functional as F
import copy
from model.transformer import Transformer,subsequent_mask
from random import randint
from einops import rearrange
def clone(module,N):
    '''copy the given module N times'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x
class PrefixLM(nn.Module):
    def __init__(
            self,
            d_model=512,
            input_resolution=224,
            patch_size=16,
            num_text_tokens=20000,
            txt_seq_len=256,
            heads=8,
            enc_depth=8,
            dec_depth=8,
            res_depth=3,
            d_ff=1024,
            dropout=0.1
    ):
        super(PrefixLM,self).__init__()
        assert input_resolution%patch_size==0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.resnet=nn.Sequential(clone(ResBlock(d_model),res_depth))
        self.txt_embed=nn.Embedding(num_text_tokens+2,d_model)#<seg>和<eof>
        self.txt_pos_embed=nn.Embedding(txt_seq_len*2,d_model)#encoder和decoder中txt长度均为256
        self.img_tokens_len=(input_resolution // patch_size) ** 2
        self.img_pos_embed=nn.Embedding(self.img_tokens_len,d_model)
        self.txt_seq_len=txt_seq_len
        self.num_text_tokens=num_text_tokens
        tgt_mask=subsequent_mask(self.txt_seq_len+1)#add <bos>
        self.register_buffer('tgt_mask', tgt_mask, persistent=False)
        self.transformer=Transformer(d_model,heads,enc_depth,dec_depth,d_ff,src_mask=None,tgt_mask=tgt_mask,dropout=dropout)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_text_tokens),
        )
    def forward(self,img,txt,return_loss=False):
        device=txt.device
        img_emed=self.conv1(img)
        img_emed=self.resnet(img_emed)
        img_emed+=self.img_pos_embed(torch.arange(self.img_tokens_len,device=device))
        seg=torch.zeros(self.txt_seq_len,device=device)+self.num_text_tokens+1#<seg>
        end=torch.zeros(self.txt_seq_len,device=device)+self.num_text_tokens+2#<end>
        l=randint(4,15)

        pre_txt=torch.zeros_like(txt)
        pre_txt[:,:l]=txt[:,:l]
        tgt_txt=torch.zeros_like(txt)
        tgt_txt[:,l:]=txt[:,l:]
        del txt,img
        pre_txt=torch.where(pre_txt==0,seg,pre_txt)
        tgt_txt=torch.where(tgt_txt==0,end,tgt_txt)
        tgt_txt= F.pad(tgt_txt, (1, 0), value = 0)#add<bos>
        labels=tgt_txt[:,1:]
        tgt_txt=tgt_txt[:,:-1]#
        del seg, end
        pre_txt_embed=self.txt_embed(pre_txt)
        pre_txt_embed+=self.txt_pos_embed(torch.arange(self.txt_seq_len,device=device))
        tgt_txt_embed=self.txt_embed(tgt_txt)
        tgt_txt_embed+=self.txt_pos_embed(torch.arange(self.txt_seq_len,device=device)+256)

        prefix=torch.cat((img_emed,pre_txt_embed),dim=1)
        out=self.transformer(prefix,tgt_txt_embed)
        logits=self.to_logits(out)
        if not return_loss:
            return logits
        logits = rearrange(logits, 'b n c -> b c n')
        loss=F.cross_entropy(logits,labels)
        return loss
