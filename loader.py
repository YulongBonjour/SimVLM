from PIL import Image
from io import BytesIO
from random import randint
import lmdb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class TextImageDataset(Dataset):
    def __init__(self,
                 lmdb_folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=True
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.env = lmdb.open(
            lmdb_folder,
            max_readers=2048,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_folder)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
            print('total samples:', self.length)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return self.length

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):

        with self.env.begin(write=False) as txn:
            try:
               imgKey='img-%d'%ind
               captionKey = 'cap-%d' % ind
               img=txn.get(imgKey.encode())
               img = Image.open(BytesIO(img))
               caption=txn.get(captionKey.encode()).decode()
            except :
                print('db error,ind:', ind)
                return self.skip_sample(ind)
        try:
            tokenized_text = self.tokenizer.tokenize(
                caption,
                self.text_len,
                truncate_text=self.truncate_captions
               ).squeeze(0)
            image_tensor = self.image_transform(img)
        except:
              print(f"Skipping index {ind}")
              return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor


if __name__=="__main__":
    ds=lmdb_Dataset('D:/Dalle_lmdb')
    loader=torch.utils.data.DataLoader(ds)
    for _ ,data in enumerate(loader):
        text,img=data
        print(text,img)

