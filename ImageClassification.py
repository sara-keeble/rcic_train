import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os.path import exists
import argparse
import numpy as np
import pandas as pd
import torchvision
import fastai
from fastai.vision import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import cv2
from efficientnet_pytorch import *
from fastai.metrics import *


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
    return Image(pil2tensor(image, np.float32).div_(255))
    
class MultiChannelImageList(ImageList):
    def open(self, fn):
        return open_rcic_image(fn)
    
def efficientnet_multichannel(pretrained=True,name='b0',num_classes=1108,num_channels=6,image_size=256):
    model = EfficientNet.from_pretrained('efficientnet-'+name,num_classes=num_classes)
    w = model._conv_stem.weight
    model._conv_stem = utils.Conv2dStaticSamePadding(num_channels,32,kernel_size=(3, 3), stride=(2, 2), bias=False, image_size = image_size)
    model._conv_stem.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    return model

def efficientnetb0(pretrained=True,num_channels=6):
    return efficientnet_multichannel(pretrained=pretrained,name='b0',num_channels=num_channels)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.train == True:
    train_df = pd.read_csv(f'{args.path}/train.csv')
    if site == 'both':
        proc_train_df_1 = generate_df(train_df, sample_num=1)
        proc_train_df_2 = generate_df(train_df, sample_num=2)
        proc_train_df = pd.concat([proc_train_df_1, proc_train_df_2])
    else:
        proc_train_df = generate_df(train_df, sample_num=site)

        proc_train_df = proc_train_df[(proc_train_df.path != 'RPE-04/Plate3/E04_s1_w') & \
                                      (proc_train_df.path != 'RPE-04/Plate3/E04_s2_w') & \
                                      (proc_train_df.path != 'HUVEC-06/Plate1/B18_s2_w') & \
                                      (proc_train_df.path != 'HUVEC-06/Plate1/B18_s1_w')]


    train_df,val_df = train_test_split(proc_train_df, test_size=args.valsplit, stratify = proc_train_df.sirna, random_state=42)
    _proc_train_df = pd.concat([train_df,val_df])
    
    data = (MultiChannelImageList.from_df(df=_proc_train_df, path=f'{args.path}/train')
        .split_by_idx(list(range(len(train_df),len(_proc_train_df))))
        .label_from_df()
        .transform(get_transforms(),size=256)
        .databunch(bs=128,num_workers=4)
        .normalize()
       )
    
    if device=='cuda':
        learn = Learner(data, efficientnetb0(),metrics=[accuracy], model_dir=args.path).to_fp16()
    else:
        learn = Learner(data, efficientnetb0(),metrics=[accuracy], model_dir=args.path).to_fp32()

    if args.loadstate is not None:
        learn = learn.load(f'{args.loadstate}', device = device)
    
    learn.unfreeze()
    learn.fit_one_cycle(epochs, lrate)

if args.export is not None:
    learn.save(Path(f"{args.path}/{args.export}"))
    learn.export(file = Path(f"{args.path}/{args.export}"))

if args.eval == True:
    test_df = pd.read_csv(f'{args.path}/test.csv')
    proc_test_df = generate_df(test_df.copy(), sample_num=1)
    proc_test_df = proc_test_df[(proc_test_df.path != 'HUVEC-18/Plate3/D23_s1_w') & (proc_test_df.path != 'RPE-09/Plate2/J16_s1_w')]

    data_test = MultiChannelImageList.from_df(df=proc_test_df, path=f'{args.path}/test')
    learn.data.add_test(data_test)

    preds, targets = learn.get_preds(DatasetType.Test)
    preds_ = preds.argmax(dim=-1)
    proc_test_df.sirna = preds_.numpy().astype(int)
    proc_test_df.to_csv('submission.csv',index=False)

    
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="working directory", action="store", dest="path", default='.')
parser.add_argument("-s", "--site", help="image site to train on", action="store", choices=[1, 2, 'both'], 
                    dest="site", default=1)
parser.add_argument("-t", "--train", help="perform training", action="store_true", dest="train", default=True)
parser.add_argument("-e", "--eval", help="evaluate model on test data and write predictions to file", 
                    action="store_true", dest="eval", default=True)
parser.add_argument("-vs", "--val_split", help="validation split fraction", action="store", dest="valsplit", 
                    default=0.035)
parser.add_argument("-e", "--epochs", help="number of training epochs", action="store", type=int, dest="epochs")
parser.add_argument("-lr", "--learnrate", help="learning rate to use in training", action="store", dest="lrate")
parser.add_argument("-ls", "--loadstate", help="file name of state dict to initialize weights", action="store", dest="statedict", default=None)
parser.add_argument("-ex", "--export", help="file name to export model state dict and pickle file after training", action="store",
                   dest="export", default=None)
args = parser.parse_args()



