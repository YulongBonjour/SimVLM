'''
   author: yulong-XJTU
'''
import torch
from torch import nn
import torch.nn.functional as F
import copy
from axial_positional_embedding import AxialPositionalEmbedding
from model.resblock import BottleneckBlock
from random import randint
from einops import rearrange
from model.torch_transformer import encoder,decoder
from torch.nn.init import xavier_uniform_
from typing import Optional,Union
def clone(module,N):
    '''copy the given module N times'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class SimVLM(nn.Module):
    def __init__(
            self,
            d_model=512,
            input_resolution=224,
            patch_size=16,
            num_text_tokens=10000,
            txt_seq_len=60,
            prefix_txt_len=20,
            target_txt_len=60,
            max_trunc_txt_len=15,
            heads=8,
            enc_depth=12,
            dec_depth=12,
            d_ff=1024,
            activation="relu",
            dropout=0.,
            pad_idx=0,
    ):
        super(SimVLM,self).__init__()
        assert input_resolution%patch_size==0 and max_trunc_txt_len<=prefix_txt_len and max_trunc_txt_len<txt_seq_len
        self.ResNet=nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=64, kernel_size=patch_size, stride=patch_size, bias=True),
                                    BottleneckBlock(in_channels=64,out_channels=256,bottleneck_channels=64,),
                                    BottleneckBlock(in_channels=256,out_channels=d_model,bottleneck_channels=128)])
        self.txt_embed = nn.Embedding(num_text_tokens, d_model)
        self.txt_pos_embed=nn.Embedding(txt_seq_len,d_model)
        image_fmap_size=input_resolution // patch_size
        self.img_tokens_len=image_fmap_size ** 2
        # self.img_pos_embed=nn.Embedding(self.img_tokens_len,d_model)
        self.img_pos_embed = AxialPositionalEmbedding(d_model, axial_shape = (image_fmap_size, image_fmap_size))
        self.txt_seq_len=txt_seq_len
        self.target_txt_len=target_txt_len
        self.prefix_txt_len = prefix_txt_len
        self.max_trunc_txt_len=max_trunc_txt_len
        self.num_text_tokens=num_text_tokens
        self.dim_embed=d_model
        self.input_resolution=input_resolution
        self.patch_size=patch_size
        self.pad_idx=pad_idx
        # self.temperature = nn.Parameter(torch.tensor(1.))#论文中没提到
        # self.transformer=Transformer(d_model,heads,enc_depth,dec_depth,d_ff,dropout=dropout)
        self.bos_emb=nn.Parameter(torch.randn(1,d_model))
        self.encoder=encoder(d_model=d_model, nhead=heads,
                                     depth=enc_depth, dim_feedforward=d_ff, dropout=dropout, activation=activation)
        self.decoder=decoder(d_model=d_model, nhead=heads, depth=dec_depth,
                             dim_feedforward=d_ff, dropout=dropout, activation=activation)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,  num_text_tokens)
        )
        self._reset_parameters()
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self,img,txt,return_loss=False):
        device=txt.device
        n=txt.shape[0]
        img_emed=self.ResNet(img)
        img_emed=rearrange(img_emed,'b c h w -> b (h w) c')
        img_emed=img_emed+self.img_pos_embed(img_emed)
        key_padding_mask=torch.zeros(n,self.img_tokens_len+self.prefix_txt_len,dtype=torch.bool,device=device)
        txt_emb=self.txt_embed(txt)+self.txt_pos_embed(torch.arange(self.txt_seq_len,device=device))#b n c

        l = randint(0,self.max_trunc_txt_len)
        pre_txt=torch.zeros((n, self.prefix_txt_len), device=device, dtype=torch.long)
        pre_txt[:,:l]=txt[:,:l]
        key_padding_mask[:,self.img_tokens_len:]=pre_txt== self.pad_idx   # [B,prefix_txt_len+img_tokens_len]

        pre_txt_emb=torch.zeros((n, self.prefix_txt_len,self.dim_embed),device=device)
        pre_txt_emb[:,:l]=txt_emb[:,:l]

        tgt_txt_emb=torch.zeros((n,self.target_txt_len,self.dim_embed),device=device)
        tgt_txt_emb[:, :(self.txt_seq_len - l)]=txt_emb[:,l:]

        labels = torch.zeros((n, self.target_txt_len), device=device, dtype=torch.long)
        labels[:, :(self.txt_seq_len - l)] = txt[:, l:]
        del txt,img
        #add<bos>
        tgt_txt_emb=torch.cat([torch.zeros(n,1,self.dim_embed,device=device)+self.bos_emb,tgt_txt_emb[:,:-1]],dim=1)
        prefix=torch.cat((img_emed,pre_txt_emb),dim=1)

        tgt_txt_emb=rearrange(tgt_txt_emb, 'b n c -> n b c')
        prefix=rearrange(prefix,'b n c -> n b c')
        tgt_mask = generate_square_subsequent_mask(self.target_txt_len).to(device)

        memory = self.encoder(prefix, mask=None, src_key_padding_mask=key_padding_mask)
        output = self.decoder(tgt_txt_emb, memory, tgt_mask=tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=key_padding_mask)
        logits=self.to_logits(output)#seq_len, batch,vocab_size
        if not return_loss:
            return logits
        # temp = self.temperature.exp()
        logits = rearrange(output, 'n b c -> b c n')
        # logits=logits*temp #带温度参数
        loss=F.cross_entropy(logits,labels,ignore_index=0)
        return loss

    def generate(self,img,prefix_txt,sampling_method='nucleus',eos_id=0,top_k=256,top_p=0.9,temperature=1.):
        device = img.device
        n = img.shape[0]
        img_emed = self.ResNet(img)
        img_emed = rearrange(img_emed, 'b c h w -> b (h w) c')
        img_emed = img_emed + self.img_pos_embed(img_emed)
        prefix_txt_len=(prefix_txt!=self.pad_idx).sum(dim=-1).unsqueeze(-1)#[B,1]
        key_padding_mask = torch.zeros(n, self.img_tokens_len + self.prefix_txt_len, dtype=torch.bool, device=device)
        key_padding_mask[:, self.img_tokens_len:] = prefix_txt == self.pad_idx
        prefix_txt_emb = self.txt_embed(prefix_txt) + \
                         self.txt_pos_embed(torch.arange(self.prefix_txt_len, device=device))
        prefix = torch.cat((img_emed, prefix_txt_emb), dim=1)
        prefix = rearrange(prefix, 'b n c -> n b c')
        memory = self.encoder(prefix, mask=None, src_key_padding_mask=key_padding_mask)

        if sampling_method=='nucleus':
            cap_tokens=self.nucleus_sampling(memory,prefix_len=prefix_txt_len,eos_id=eos_id,
                                             memory_key_padding_mask=key_padding_mask,
                                             top_k=top_k,
                                             top_p=top_p,
                                             temperature=temperature)
        elif sampling_method=='greedy':
            cap_tokens=self.sampling(memory,prefix_len=prefix_txt_len,
                                     eos_id=eos_id,
                                     memory_key_padding_mask=key_padding_mask,
                                     mode='greedy')
        else:
            cap_tokens = self.sampling(memory, prefix_len=prefix_txt_len, eos_id=eos_id,
                                       memory_key_padding_mask=key_padding_mask,
                                       mode='random')
        return cap_tokens
    def core(self,tgt, memory,memory_key_padding_mask):
            '''
            :param memory:   n b d
            :param memory_key_padding_mask:   b n
            :return:  logits for next token
            '''
            out = self.decoder(tgt, memory, tgt_mask=None, memory_mask=None,
                                  tgt_key_padding_mask=None,
                                  memory_key_padding_mask=memory_key_padding_mask)
            # logits = self.to_logits(output)  # seq_len, batch,vocab_size
            return out[-1, :] # [B, D]

    def get_logprobs_state(self,memory, tgt_emb,memory_key_padding_mask, output_logsoftmax=1):
            output = self.core(tgt_emb, memory,memory_key_padding_mask=memory_key_padding_mask)
            if output_logsoftmax==1:
                logprobs = F.log_softmax(self.to_logits(output), dim=1)
            else:
                logprobs = self.to_logits(output)
            return logprobs #[B,Vocab]

    def sampling(self,memory,prefix_len,eos_id,memory_key_padding_mask,
                   return_logprobs=False,mode='greedy'):
            b, device = memory.shape[1], memory.device
            seq = torch.zeros(b, self.target_txt_len, dtype=torch.long).to(device)
            seqlogprobs = torch.zeros(b, self.target_txt_len, self.num_text_tokens).to(device)
            done = torch.tensor([False for _ in range(b)], device=device)#[B]
            cap = torch.tensor([[]] * b, dtype=torch.long, device=device)  # [B 1]
            cap_embs=torch.zeros(b,1,self.dim_embed,device=device)+self.bos_emb# b 1 d
            cap_embs = rearrange(cap_embs, 'b n c -> n b c')
            cur_len = 0
            trigrams = []
            max_pre_len=max(prefix_len)
            while (cur_len < self.txt_seq_len-max_pre_len):
                logprobs = self.get_logprobs_state(memory, cap_embs, memory_key_padding_mask=memory_key_padding_mask,
                                                   output_logsoftmax=1)  # B V
                # Mess with trigrams
                # Copy from https://github.com/lukemelas/image-paragraph-captioning
                if cur_len >= 3:
                    # Store trigram generated at last step
                    prev_two_batch = cap[:, cur_len - 3:cur_len - 1]
                    for i in range(b):  # = seq.size(0)
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        current = cap[i][cur_len - 1].item()
                        if cur_len == 3:  # initialize
                            trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                        elif cur_len > 3:
                            if prev_two in trigrams[i]:  # add to list
                                trigrams[i][prev_two].append(current)
                            else:  # create list
                                trigrams[i][prev_two] = [current]
                    # Block used trigrams at next step
                    prev_two_batch = cap[:, cur_len - 2:cur_len]
                    mask = torch.zeros(logprobs.size(), requires_grad=False).to(
                        logprobs.device)  # batch_size x vocab_size
                    for i in range(b):
                        prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                        if prev_two in trigrams[i]:
                            for j in trigrams[i][prev_two]:
                                mask[i, j] += 1
                    # print(mask)
                    # Apply mask to log probs
                    # logprobs = logprobs - (mask * 1e9)
                    alpha = 2.0  # = 4
                    logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
                # print(trigrams)
                seqlogprobs[:, cur_len] = logprobs
                if mode=='greedy':
                    sample = torch.argmax(logprobs, dim=-1)  # [B]
                else:
                    sample = torch.distributions.Categorical(logits=logprobs.detach()).sample()
                sample[done] = self.pad_idx
                is_done = sample == eos_id
                sample=sample.unsqueeze(-1)#[B 1]
                cap = torch.cat((cap, sample), dim=-1)
                new_cap_embs=self.txt_embed(sample)+self.txt_pos_embed(prefix_len+cur_len)#[B 1,D]
                new_cap_embs = rearrange(new_cap_embs, 'b n c -> n b c')
                cap_embs=torch.cat([cap_embs,new_cap_embs],dim=0)
                done += is_done
                cur_len += 1
                all_done = False not in done
                if all_done: break
            seq[:, :cur_len] = cap[:, :]
            if return_logprobs:
                return seq, seqlogprobs
            else:
                return seq


    def nucleus_sampling(self, memory,prefix_len,eos_id,memory_key_padding_mask, top_k, top_p, temperature):
            '''
            prefix_len: [B 1]
            '''
            # logit
            b, device = memory.shape[1], memory.device
            seq = torch.zeros(b, self.target_txt_len, dtype=torch.long).to(device)
            done = torch.tensor([False for _ in range(b)], device=device)
            cap = torch.tensor([[]] * b, dtype=torch.long, device=device)  # [B 1]
            cap_embs = torch.zeros(b, 1, self.dim_embed, device=device) + self.bos_emb  # b 1 d
            cap_embs = rearrange(cap_embs, 'b n c -> n b c')
            cur_len = 0
            max_pre_len = max(prefix_len)
            while (cur_len < self.txt_seq_len-max_pre_len):
                logit = self.get_logprobs_state(memory, cap_embs, memory_key_padding_mask=memory_key_padding_mask,
                                                   output_logsoftmax=0) / temperature
                probs = self.top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p, device=device)
                sample = torch.multinomial(probs, 1)[:,0]  # [B 1]
                sample[done] = self.pad_idx
                is_done = sample == eos_id
                sample = sample.unsqueeze(-1)  # [B 1]
                cap = torch.cat((cap, sample), dim=-1)
                new_cap_embs = self.txt_embed(sample) + self.txt_pos_embed(prefix_len + cur_len)  # [B 1,D]
                new_cap_embs = rearrange(new_cap_embs, 'b n c -> n b c')
                cap_embs = torch.cat([cap_embs, new_cap_embs], dim=0)
                done += is_done
                cur_len += 1
                all_done = False not in done
                if all_done: break
            seq[:, :cur_len] = cap[:, :]
            return seq

    def top_k_top_p_filtering(self,
                                  next_token_logits: torch.FloatTensor,
                                  top_k: Optional[float] = None,
                                  top_p: Optional[float] = None,
                                  device: Union[str, torch.device] = "cpu",
                                  ) -> torch.FloatTensor:
            if top_k is None:
                top_k = next_token_logits.shape[-1]
            if top_p is None:
                top_p = 1.0
            p, largest_p_idx = F.softmax(next_token_logits, dim=-1).topk(top_k, dim=-1)
            cumulative_p = p.cumsum(dim=-1)
            threshold_repeated = top_p + torch.zeros((len(p), 1)).to(device)
            idx = torch.searchsorted(cumulative_p, threshold_repeated).clip(max=top_k - 1).squeeze()
            cutoffs = cumulative_p[torch.arange(len(cumulative_p)), idx]
            censored_p = (cumulative_p <= cutoffs[:, None]) * p
            renormalized_p = censored_p / censored_p.sum(dim=-1, keepdims=True)
            final_p = torch.zeros_like(next_token_logits)
            row_idx = torch.arange(len(p)).unsqueeze(1).repeat(1, top_k).to(device)
            final_p[row_idx, largest_p_idx] = renormalized_p.to(final_p.dtype)
            return final_p

