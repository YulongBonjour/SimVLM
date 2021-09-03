# -*- coding:utf-8 -*-

import os
import lmdb
from pathlib import Path
import cv2
import glob
import numpy as np
import PIL
from random import choice
from PIL import ImageFile
from tqdm import *

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, captionPathList, map_size=None):
    """
    Create LMDB dataset for DALLE training.
#    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        captionpathList     : list of corresponding groundtruth text
    """

    assert (len(imagePathList) == len(captionPathList))
    nSamples = len(imagePathList)
    print('...................')
    env = lmdb.open(outputPath)
    if map_size is not None:
        env.set_mapsize(map_size)
    with env.begin(write=False) as txn:
        try:
            cnt = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
        except:
            cnt = 0

    cache = {}
    for i in range(nSamples):
        # print(cnt)
        imagePath = imagePathList[i]
        captionPath = captionPathList[i]
        if not os.path.exists(imagePath) or not os.path.exists(captionPath):
            continue
        try:
            # image= PIL.Image.open(imagePath)
            with open(imagePath, 'rb') as f:
                image = f.read()
        except:
            print('A error occured while loading %s' % imagePath)
            continue

        print(captionPath)
        try:
            f = open(captionPath, 'r', encoding='utf-8')
            descriptions = f.readlines()
            f.close()
        except:
            continue
        try:
            # descriptions = captionPath.read_text(encoding='utf-8').split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {captionPath}.")
            continue

        # cache[str(cnt).encode()]=(image,description)

        imageKey = 'img-%d' % cnt
        captionKey = 'cap-%d' % cnt
        print(imageKey)
        cache[imageKey.encode()] = image
        cache[captionKey.encode()] = description.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

    env.close()


###########################  请使用下面的函数  ##########################################
def generate_or_add_lmdb(outputPath, raw_data_folder, map_size):
    '''
        将raw_data_folder中的 原始数据  加到outputPath的  lmdb数据库 中
        请确保不要重复添加
    '''
    text_files_path = Path(raw_data_folder).joinpath('txt')
    image_files_path = Path(raw_data_folder).joinpath('img')
    text_files = [*text_files_path.glob('*.txt')]
    image_files = [*image_files_path.glob('*')]
    '''
        *image_files_path.glob('*.png'), *image_files_path.glob('*.jpg'),
        *image_files_path.glob('*.jpeg'), *image_files_path.glob('*.bmp')
    ]'''

    text_files = {text_file.stem: text_file for text_file in text_files}
    image_files = {image_file.stem: image_file for image_file in image_files}
    imgPaths = []
    captionPathList = []
    keys = (image_files.keys() & text_files.keys())
    for key in keys:
        imgPaths.append(image_files[key])
        captionPathList.append(text_files[key])

    createDataset(outputPath, imgPaths, captionPathList, map_size)


def merge_lmdbs(outputPath, pathList, map_size=None):
    '''
       合并 多个 lmdb到一个 新的lmdb（可以是之前存在的）
       请确保不要重复合并、添加
    '''

    env_prim = lmdb.open(outputPath)
    if map_size is not None:
        env_prim.set_mapsize(map_size)
    with env_prim.begin(write=False) as txn:
        try:
            cnt = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
        except:
            cnt = 0
    for db_path in pathList:
        if (outputPath == db_path):
            print('合并对象不能是目标输出数据库')
            continue
        if not os.path.exists(db_path):
            print(db_path, ": 不存在")
            continue
        env_curr = lmdb.open(db_path)
        print('adding ', db_path)
        with env_prim.begin(write=True) as txn1:
            with env_curr.begin(write=False) as txn2:
                for ind in tqdm(range(int(txn2.get('num-samples'.encode('utf-8')).decode('utf-8')))):
                    imgKey = 'img-%d' % ind
                    capKey = 'cap-%d' % ind
                    img = txn2.get(imgKey.encode('utf-8'))
                    cap = txn2.get(capKey.encode('utf-8'))
                    imgKey = 'img-%d' % cnt
                    capKey = 'cap-%d' % cnt
                    txn1.put(imgKey.encode(), img)
                    txn1.put(capKey.encode(), cap)
                    cnt += 1
            txn1.put('num-samples'.encode(), str(cnt).encode())

        env_curr.close()
        print(cnt)

    env_prim.close()
def add_part_lmdb(outputlmdb,sourceLmdb,start=0,map_size=None):
    '''
    将sourcelmdb中从index为start起的图文对加入outputlmdb中
    '''
    if (outputlmdb == sourceLmdb):
        print('合并对象不能是目标输出数据库')
        return
    env_prim = lmdb.open(outputlmdb)
    if map_size is not None:
        env_prim.set_mapsize(map_size)
    env_source=lmdb.open(sourceLmdb)
    with env_source.begin(write=False) as txn_s:
        cnt_s = int(txn_s.get('num-samples'.encode('utf-8')).decode('utf-8'))
        assert cnt_s>start
        with env_prim.begin(write=False) as txn:
            try:
                cnt = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
            except:
                cnt = 0
            for offset  in tqdm(range(start,cnt_s)):
                imgKey = 'img-%d' % offset
                capKey = 'cap-%d' % offset
                img = txn_s.get(imgKey.encode('utf-8'))
                cap = txn_s.get(capKey.encode('utf-8'))
                imgKey = 'img-%d' % cnt
                capKey = 'cap-%d' % cnt
                txn.put(imgKey.encode(), img)
                txn.put(capKey.encode(), cap)
                cnt += 1
            txn.put('num-samples'.encode(), str(cnt).encode())
    env_prim.close()
    env_source.close()



if __name__ == '__main__':
    #1，不要重复添加
    #2，merge_lmdbs()速度很快，推荐使用

    outputPath1 = 'D:/Dalle_lmdb'
    outputPath2 = 'D:/Dalle_lmdb1'
    outputPath3='D:/Dalle_lmdb2'
    folder= "D:/2222"
   # generate_or_add_lmdb(outputPath1,folder,824288000)
    #merge_lmdbs('D:/new_2222_lmdb',[outputPath2,outputPath1,outputPath3],1824288000)
    env=lmdb.open('D:/new_2222_lmdb')
    with env.begin(write=False) as txn:
        print(int(txn.get('num-samples'.encode('utf-8')).decode('utf-8')))
    env.close()
