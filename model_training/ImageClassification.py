import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from os.path import exists
import argparse
import numpy as np
import pandas as pd
import fastai
from fastai.vision import *


def generate_df(train_df,sample_num=1):
    """Generates file paths for training images from info in train.csv"""
    train_df['wpath'] = train_df['plate'].astype(str).str.cat(train_df['well'], sep='/')
    train_df['path'] = train_df['experiment'].str.cat(train_df['wpath'] ,sep='/Plate')
    train_df['path'] = train_df['path'] + f'_s{str(sample_num)}_w'
    train_df = train_df.drop(columns=['id_code','experiment','plate','well', 'wpath']).reindex(columns=['path','sirna'])
    return train_df


def open_rcic_image(fn):
    """Open six-channel image from separate .png files"""
    images = []
    for i in range(6):
        file_name = fn+str(i+1)+'.png'
        assert exists(file_name),f"File does not exist: {file_name}"
        im = cv2.imread(file_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)
    image = np.dstack(images)
    return Image(pil2tensor(image, np.float32).div_(255))
    
    
class MultiChannelImageList(ImageList):
    """Extension of Fastai's ImageList to use custom open function"""
    def open(self, fn):
        return open_rcic_image(fn)
    
    
def efficientnet_multichannel(pretrained=True,name='b0',num_classes=1108,num_channels=6,image_size=256):
    """Alters efficientnet model to input 6-channel images"""
    model = EfficientNet.from_pretrained('efficientnet-'+name,num_classes=num_classes)
    w = model._conv_stem.weight
    model._conv_stem = utils.Conv2dStaticSamePadding(num_channels,32,kernel_size=(3, 3), stride=(2, 2), 
                                                     bias=False, image_size = image_size)
    #Using an average of existing weights for new input channels
    model._conv_stem.weight = nn.Parameter(torch.stack([torch.mean(w, 1)]*num_channels, dim=1))
    return model


def efficientnetb0(pretrained=True,num_channels=6):
    """Get pretrained model with 6-channel image input"""
    return efficientnet_multichannel(pretrained=pretrained,name='b0',num_channels=num_channels)

      
def train_model(path, site, statedict, epochs, lrate):
    """Train the efficientnet model with pretrained initialization weights"""
    
    train_df = pd.read_csv(f'{path}/train.csv')
    #generate list of file paths
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
    
    #split train dataset into validation and training
    train_df,val_df = train_test_split(proc_train_df, test_size=0.035, stratify = proc_train_df.sirna, random_state=42)
    final_train_df = pd.concat([train_df,val_df])
    
    #process the images
    data = (MultiChannelImageList.from_df(df=final_train_df, path=f'{path}/train')
        .split_by_idx(list(range(len(train_df),len(final_train_df))))
        .label_from_df()
        .transform(get_transforms(),size=256)
        .databunch(bs=128,num_workers=4)
        .normalize()
       )
    if device=='cuda':
        learn = Learner(data, efficientnetb0(),metrics=[accuracy], model_dir=path).to_fp16()
    else:
        learn = Learner(data, efficientnetb0(),metrics=[accuracy], model_dir=path).to_fp32()
    
    #Load in prior weights if continuing from intermediate checkpoint
    if statedict is not None:
        learn = learn.load(f'{statedict}', device = device)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, lrate)

    
def make_test_predictions(path):
    """Loads test dataset and makes predictions of siRNA treatment"""
    test_df = pd.read_csv(f'{path}/test.csv')
    proc_test_df = generate_df(test_df.copy(), sample_num=1)
    #some samples didn't exist in the dataset
    proc_test_df = proc_test_df[(proc_test_df.path != 'HUVEC-18/Plate3/D23_s1_w') & (proc_test_df.path != 'RPE-09/Plate2/J16_s1_w')]

    data_test = MultiChannelImageList.from_df(df=proc_test_df, path=f'{path}/test')
    learn.data.add_test(data_test)

    preds, targets = learn.get_preds(DatasetType.Test)
    preds_ = preds.argmax(dim=-1)
    proc_test_df.sirna = preds_.numpy().astype(int)
    proc_test_df.to_csv('test_predictions.csv',index=False)


    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="working directory", action="store", dest="path", default='.')
    parser.add_argument("-s", "--site", help="image site to train on", action="store", choices=['1', '2', 'both'], 
                        dest="site", default=1)
    parser.add_argument("-t", "--train", help="perform training", action="store_true", dest="train", default=True)
    parser.add_argument("-ev", "--eval", help="evaluate model on test data and write predictions to file", 
                        action="store_true", dest="eval", default=True)
    parser.add_argument("-e", "--epochs", help="number of training epochs", action="store", type=int, dest="epochs")
    parser.add_argument("-lr", "--learnrate", help="learning rate to use in training", action="store", type=float, dest="lrate")
    parser.add_argument("-ls", "--loadstate", help="file name of state dict to initialize weights", action="store", dest="statedict")
    parser.add_argument("-ex", "--export", help="file name to export model state dict and pickle file after training", action="store",
                       dest="export")

    args = parser.parse_args()
    
    import torchvision

    import cv2
    from efficientnet_pytorch import *
    from fastai.metrics import *
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split
    
    print("Modules imported")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.train == True:
        print("Training model")
    
        train_model(args.path, args.site, args.statedict, args.epochs, args.lrate)

        print("Training complete")
        if args.export is not None:
            learn.save(Path(f"{args.path}/{args.export}"))
            learn.export(file = Path(f"{args.path}/{args.export}"))

    if args.eval == True:
        print("Making predictions")
        make_test_predictions(args.path)

