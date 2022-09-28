from model import PrefixLM,ChineseTokenizer
import torch
import argparse
import os
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from einops import rearrange

'''parser=argparse.ArgumentParser()
parser.add_argument('--prefixLM_path',type=str,help='path to your trained PrefixLM')
parser.add_argument('--img_path',type=str,help='path to your images')
parser.add_argument('--prefix',type=str,default='',help='prefix for caption')

args=parser.parse_args()'''

device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

prefixLM_path='./SimVLP2.pt'

def generate_caption(model, image_tensor, tokenized_text):
    img_emed = model.ResNet(image_tensor)
    img_emed = rearrange(img_emed, 'b c h w -> b (h w) c')
    img_emed += model.img_pos_embed(img_emed)
    # seg = torch.arange(model.prefix_txt_len, device=device, dtype=torch.long) + model.num_text_tokens  # position tokens
    #seg = torch.zeros(model.txt_seq_len, device=device, dtype=torch.long) + model.num_text_tokens + 1  # <seg>
    # pre_txt = torch.where(tokenized_text == 0, seg, tokenized_text)

    pre_txt_embed = model.txt_embed(tokenized_text)
    pre_txt_embed += model.txt_pos_embed(torch.arange(model.prefix_txt_len, device=device))
    tgt_txt = torch.zeros(1, 1, dtype=torch.long, device=device)+4
    tgt_txt_embed = model.txt_embed(tgt_txt)
    tgt_txt_embed += model.txt_pos_embed(torch.arange(1, device=device) + model.prefix_txt_len)

    prefix = torch.cat((img_emed, pre_txt_embed), dim=1)
    out = model.transformer(prefix, tgt_txt_embed)
    logits = model.to_logits(out)[:, -1]
    #logits=logits[:,:-26]
    sample = torch.argmax(logits, dim=-1)
    cur_len = 1
    while (cur_len < model.target_txt_len and sample!=5):
        tgt_txt = torch.cat((tgt_txt, sample.unsqueeze(1)), dim=-1)
        tgt_txt_embed = model.txt_embed(tgt_txt)
        cur_len += 1
        tgt_txt_embed += model.txt_pos_embed(torch.arange(cur_len, device=device) + model.prefix_txt_len)
        out = model.transformer(prefix, tgt_txt_embed)
        logits = model.to_logits(out)[:, -1]
        #logits = logits[:, :-26]
        #print(logits)
        sample = torch.argmax(logits, dim=-1)
    return tgt_txt

assert os.path.exists(prefixLM_path), 'trained model path must exist'
loaded_obj = torch.load(prefixLM_path, map_location='cpu')
PrefixLM_configure, weights = loaded_obj['hparams'],loaded_obj['weights']
model=PrefixLM(**PrefixLM_configure)
model.load_state_dict(weights)

model.to(device)
image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(model.input_resolution,
                                scale=(0.75, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])
tokenizer=ChineseTokenizer()

'''tokenized_text = tokenizer.tokenize(
            args.prefix,
            model.prefix_txt_len,
            truncate_text=True
        ).to(device)
#print(tokenized_text)
img=PIL.Image.open(args.img_path)
image_tensor = image_transform(img).unsqueeze(0).to(device)
model.eval()
cap=generate_caption(model, image_tensor, tokenized_text)
#print(cap)
print(args.prefix+tokenizer.decode(cap.squeeze(0)).replace('[UNK]',''))'''

def interface(image_path="C:/Users/17914/Pictures/Camera Roll/WIN_20210207_23_12_14_Pro.jpg",prefix=''):
    img = PIL.Image.open(image_path)
    image_tensor = image_transform(img).unsqueeze(0).to(device)
    tokenized_text = tokenizer.tokenize(
        prefix,
        model.prefix_txt_len,
        truncate_text=True,
        train=False
    ).to(device)
    model.eval()
    cap = generate_caption(model, image_tensor, tokenized_text)
    return prefix + tokenizer.decode(cap.squeeze(0)).replace('[UNK]', '')

if __name__=='__main__':
    print(interface("C:\\Users\\17914\Pictures\\Camera Roll\\WIN_20210207_23_12_14_Pro.jpg"))
