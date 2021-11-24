from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastai.vision import *
import cv2
import numpy
from efficientnet_pytorch import *
import torchvision
import train_cnn 
from train_cnn import MultiChannelImageList

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

def open_rcic_image(filelist):
    images = []
    for file_name in filelist:
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

class ModelPredict():
    def __init__(self):
        #train_cnn.main()
        empty_data = ImageDataBunch.load_empty(path='.', fname='databunch_ada19_export.pkl')        
        learn = Learner(empty_data, efficientnetb0(), metrics=[accuracy], model_dir='.')
        learn = learn.load('efficientnet_14_epochs_site2_export', device = 'cpu').to_fp32()
    
    def predict(self, files):
        self.filenames = files
        img = open_rcic_image(self.filenames)
        pred_class, pred_idx, outputs = learn.predict(img)
        return str(pred_class)
    
m = ModelPredict()

@app.get("/")
def main():
    content = """
<body>
<form action="/predict/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.post("/predict/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    fnames = []
    for file in files:
        contents = await file.read()
        filename = 'static/' + file.filename
        with open(filename, 'wb') as f:
            f.write(contents)
        fnames.append(filename)
    pred = m.predict(fnames)
    return pred

