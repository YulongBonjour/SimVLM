# SimVLM
pytorch implementation for SimVLM ---SIMPLE VISUAL LANGUAGE MODEL PRETRAINING WITH WEAK SUPERVISION
https://arxiv.org/abs/2108.10904

The tokenizer used in current codes is ChineseTokenzier, if you change it to anther tokenizer, some modifications are needed.
* in prefixLM.py: please locate this annotation  #add[CLS], if you change the tokenizer, don't forget  to change the token ID. another [SEP] token is added at the ending(in the tokenizer.py,please check.)
* in generated_caption.py, line 36, 4 is the the ID of the [start of sentence token]; And line 46, 5 is the id of [SEP]
* in tokenizer.py  please locate this annotation  #special token: [CLS]==4,[SEP]==5, [PAD]==0,[bos]=7

# datasets format
    for 'loader.py' ,we use lmdb datasets
    for 'loader_ori.py', the format of dataset is like this:
           --------
                   |___txts
                   |   |__id1.txt
                   |   |__id2.txt
                   |   ....
                   |___imgs
                   |   |__id1.jpg
                   |   |__id2.jpg
                   |   ......
XXX.txt may contain multiple captions for XXX.jpg, each caption occupies a line. In every epoch, we randomly choose a caption from the candidates for each image. 

you can also change the loader to any other form according to your need. For example, if your dataset is small, you can gather all the captions to a single file(eg.json) and get the texts acocording to image ids, and open the image using their stored paths.
