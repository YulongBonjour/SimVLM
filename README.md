# SimVLM
unofficial pytorch implementation for SimVLM ---SIMPLE VISUAL LANGUAGE MODEL PRETRAINING WITH WEAK SUPERVISION
https://arxiv.org/abs/2108.10904

* The tokenizer used in current codes is ChineseTokenzier, if you change it to anther tokenizer, some modifications are needed.
* Nucleus sampling, random sampling, and greedy samping are available.
* I would be apprecaited for any Feedback. 
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
