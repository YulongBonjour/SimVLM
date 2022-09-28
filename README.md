# SimVLM
pytorch implementation for SimVLM ---SIMPLE VISUAL LANGUAGE MODEL PRETRAINING WITH WEAK SUPERVISION
https://arxiv.org/abs/2108.10904

The tokenizer used in current codes is ChineseTokenzier, if you change it to anther tokenizer, some modifications are needed.
* in prefixLM.py: please locate this annotation  #add[CLS], if you change the tokenizer, don't forget  to change the token ID. another [SEP] token is added at the ending(in the tokenizer.py,please check.)
* in generated_caption.py, line 36, the ID of the [start of sentence token]
* in tokenizer.py  please locate this annotation  #special token: [CLS]==4,[SEP]==5, [PAD]==0,[bos]=7

# datasets format
    for 'loader.py' ,we use lmdb datasets
    for 'loader_ori.py', the format of dataset is like this:
           --------
                   |___txts
                   |___imgs
   
