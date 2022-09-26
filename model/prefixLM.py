'''
   author: yulong-XJTU
'''
import torch
from torch import nn
import torch.nn.functional as F
import copy
from model.transformer import Transformer,subsequent_mask
from axial_positional_embedding import AxialPositionalEmbedding
from model.resblock import BottleneckBlock
from random import randint
from einops import rearrange
def clone(module,N):
    '''copy the given module N times'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PrefixLM(nn.Module):
    def __init__(
            self,
            d_model=512,
            input_resolution=224,
            patch_size=16,
            num_text_tokens=10000,
            txt_seq_len=256,
            prefix_txt_len=25,
            target_txt_len=52,
            max_trunc_txt_len=15,
            heads=8,
            enc_depth=12,
            dec_depth=12,
            d_ff=1024,
            dropout=0.
    ):
        super(PrefixLM,self).__init__()
        assert input_resolution%patch_size==0 and max_trunc_txt_len<=prefix_txt_len and max_trunc_txt_len<txt_seq_len
        self.ResNet=nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=64, kernel_size=patch_size, stride=patch_size, bias=True),
                                    BottleneckBlock(in_channels=64,out_channels=256,bottleneck_channels=64,),
                                    BottleneckBlock(in_channels=256,out_channels=d_model,bottleneck_channels=128)])
        self.txt_embed = nn.Embedding( num_text_tokens, d_model)
        self.txt_pos_embed=nn.Embedding(prefix_txt_len+target_txt_len,d_model)
        image_fmap_size=input_resolution // patch_size
        self.img_tokens_len=image_fmap_size ** 2
        # self.img_pos_embed=nn.Embedding(self.img_tokens_len,d_model)
        self.img_pos_embed =  AxialPositionalEmbedding(d_model, axial_shape = (image_fmap_size, image_fmap_size))
        self.txt_seq_len=txt_seq_len
        self.target_txt_len=target_txt_len
        self.prefix_txt_len = prefix_txt_len
        self.max_trunc_txt_len=max_trunc_txt_len
        self.num_text_tokens=num_text_tokens
        self.dim_embed=d_model
        self.input_resolution=input_resolution
        self.patch_size=patch_size
        # self.temperature = nn.Parameter(torch.tensor(1.))#论文中没提到
        self.transformer=Transformer(d_model,heads,enc_depth,dec_depth,d_ff,dropout=dropout)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,  num_text_tokens)
        )
    def forward(self,img,txt,return_loss=False):
        device=txt.device
        n=txt.shape[0]
        img_emed=self.ResNet(img)
        img_emed=rearrange(img_emed,'b c h w -> b (h w) c')
        img_emed=img_emed+self.img_pos_embed(img_emed)
        l = randint(0,self.max_trunc_txt_len)
        pre_txt=torch.zeros((n, self.prefix_txt_len),device=device,dtype=torch.long)
        pre_txt[:,:l]=txt[:,:l]
        tgt_txt=torch.zeros((n,self.target_txt_len),device=device,dtype=torch.long)
        tgt_txt[:,:(self.txt_seq_len-l)]=txt[:,l:]
        del txt,img

        tgt_txt= F.pad(tgt_txt, (1, 0), value = 4)#add<CLS>
        labels=tgt_txt[:,1:]
        tgt_txt=tgt_txt[:,:-1]#

        pre_txt_embed=self.txt_embed(pre_txt)
        pre_txt_embed=pre_txt_embed+self.txt_pos_embed(torch.arange(self.prefix_txt_len,device=device))
        tgt_txt_embed=self.txt_embed(tgt_txt)
        tgt_txt_embed=tgt_txt_embed+self.txt_pos_embed(torch.arange(self.target_txt_len,device=device)+self.prefix_txt_len)

        prefix=torch.cat((img_emed,pre_txt_embed),dim=1)
        tgt_mask=subsequent_mask(self.target_txt_len).to(device)
        out=self.transformer(prefix,tgt_txt_embed,tgt_mask=tgt_mask)
        logits=self.to_logits(out)
        if not return_loss:
            return logits
        # temp = self.temperature.exp()
        logits = rearrange(logits, 'b n c -> b c n')
        # logits=logits*temp #带温度参数
        loss=F.cross_entropy(logits,labels,ignore_index=0)
        return loss
