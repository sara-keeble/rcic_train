import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import torchvision
import fastai
from fastai.vision import *
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def generate_df(train_df,sample_num=1):
    train_df['path'] = train_df['experiment'].str.cat(train_df['plate'].astype(str).str.cat(train_df['well'],sep='/'),sep='/Plate') + '_s'+str(sample_num) + '_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well']).reindex(columns=['path','sirna'])
    return train_df

def open_rcic_image(fn):
    images = []
    for i in range(6):
        file_name = fn+str(i+1)+'.png'
        im = cv2.imread(file_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)
    image = np.dstack(images)
    #print(pil2tensor(image, np.float32).shape)#.div_(255).shape)
    return Image(pil2tensor(image, np.float32).div_(255))
    
class MultiChannelImageList(ImageList):
    def open(self, fn):
        return open_rcic_image(fn)

def main():
    train_df = pd.read_csv('train.csv')
    proc_train_df = generate_df(train_df, sample_num=1)
    
    train_df,val_df = train_test_split(proc_train_df,test_size=0.035, stratify = proc_train_df.sirna, random_state=42)
    _proc_train_df = pd.concat([train_df,val_df])
    
    data = (MultiChannelImageList.from_df(df=_proc_train_df,path='train')
            .split_by_idx(list(range(len(train_df),len(_proc_train_df))))
            .label_from_df()
            .transform(get_transforms(),size=256)
            .databunch(bs=128,num_workers=4)
            .normalize()
           )
    
    data.path = Path('.')
    data.export(file='databunch_ada19_export.pkl')

if __name__ == '__main__':
    main()
