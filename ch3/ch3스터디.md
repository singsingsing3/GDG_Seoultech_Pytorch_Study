## 2. 🔎SSD
### ✏️2.1 물체 감지
- 한 장의 사진에 포함된 여러 물체에 대해 영역과 이름을 확인하는 작업
- 출력
1. BBox(Bounding Box)의 위치와 크기 정보
2. BBox가 어떤 물체인지 나타내는 label
3. confidence(신뢰도)
- 라벨 정보는 감지하려는 물체의 클래스 수 + 1(배경 class)
### <SSD 흐름>
- 이 책에선 VOC dataset 활용
- SSD300 채택(이미지 크기 300x300)
- BBOX의 정보를 출력하는 것이 아닌, DBox(default box, 일반적 사각형 box)를 어떻게 변형시켜야 하는지에 대한 정보를 출력

1. 300x300 image resize

2. default box 8732개 준비

3. SSD에 image 입력

4. 신뢰도 높은 DBox 추출

5. offeset정보로 수정 및 중복 제거

6. 일정 신뢰도 이상을 최종 출력으로 선정


### ✏️2.2 DataSet 구현
- Annotation Data: 물체 위치와 라벨을 나타내는 BBox
-> **imgae와 함께 고려하여 처리해야한다**

#### 💬 1.파일 경로 리스트를 작성해보자
```
import os.path as osp
import random

import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data



%matplotlib inline

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
def make_datapath_list(rootpath):
  """
  데이터 경로를 저장한 리스트 작성

  Parameters
  ----------
  rootpath:str
      데이터 폴더 경로

  Returns
  -------
  ret: train_img_list, train_anno_list, val_img_list, val_anno_list
      데이터 경로를 저장한 리스트
  """
  # 화상 파일과 어노테이션 파일의 경로 템플릿 작성
  imagepath_template = osp.join(rootpath+'JPEGImages', '%s.jpg')
  annopath_template= osp.join(rootpath+'Annotations', '%s.xml')

  #훈련 및 검증 파일 ID(파일이름) 취득
  train_id_names=osp.join(rootpath+'ImageSets/Main/train.txt')
  val_id_names=osp.join(rootpath+'ImageSets/Main/val.txt')

  #훈련 데이터의 화상 파일과 어노테이션 파일의 경로 리스트 작성
  train_img_list=list()
  train_anno_list=list()

  for line in open(train_id_names):
    file_id=line.strip() #공백과 줄바꿈 제거
    img_path=(imagepath_template % file_id)
    anno_path=(annopath_template % file_id)
    train_img_list.append(img_path) # 경로 리스트에 추가
    train_anno_list.append(anno_path)

# 검증 데이터의 화상파일과 어노테이션 파일의 경로 리스트 작성
  val_img_list=list()
  val_anno_list=list()

  for line in open(val_id_names):
    file_id=line.strip() #공백과 줄바꿈 제거
    img_path=(imagepath_template % file_id)
    anno_path=(annopath_template % file_id)
    val_img_list.append(img_path) # 경로 리스트에 추가
    val_anno_list.append(anno_path)

  return train_img_list, train_anno_list, val_img_list, val_anno_list
```
동작이 잘 되는지 찍어보자
```
# 파일 경로 리스트 작성
rootpath='/content/drive/MyDrive/Colab Notebooks/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/'
train_img_list,train_anno_list,val_img_list,val_anno_list=make_datapath_list(rootpath)

#동작 확인
print(train_img_list[0])
```
> /content/drive/MyDrive/Colab Notebooks/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/JPEGImages/2008_000008.jpg

경로가 잘 출력 된다

#### 💬 2.Annotation data를 list로 담아내자

이제 Annotation data를 list로 변환해보자
Anno data는 xml파일로 주어졌으며
주어진 Anno Data 내부는 아래와 같다.
```
<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>174</xmin>
			<ymin>101</ymin>
			<xmax>349</xmax>
			<ymax>351</ymax>
		</bndbox>
		<part>
			<name>head</name>
			<bndbox>
				<xmin>169</xmin>
				<ymin>104</ymin>
				<xmax>209</xmax>
				<ymax>146</ymax>
			</bndbox>
		</part>
```
object에 대한 설명과 BBox정보가 담겨있다
이제 이 xml파일에 있는 정보를 list에 담아내자
```
import xml.etree.ElementTree as ET
import numpy as np
# XML 형식의 어노테이션을 리스트 형식으로 반환하는  클래스
class Anno_xml2list(object):
    """
    한 화상의 XML 형식 어노테이션 데이터를 화상 크기로 normalization하여 리스트 형식으로 반환

    Attributes
    ---------
    classes: 리스트
        VOC의 class명을 저장한 리스트
    """
    def __init__(self,classes):
        self.classes=classes

    def __call__(self, xml_path, width, height):
        """
        한 화상의 XML 형식 어노테이션 데이터를 화상 크기로 normalization하여 리스트 형식으로 반환

        Parameters
        ----------
        xml_path:str
            xml 파일 경로
        width: int
            대상 화상 폭
        height: int
            대상 화상 높이

        Returns
        ----------
        ret: [[xmin, ymin, xmax, ymax,label_ind], ...]
            물체의 어노테이션 데이터를 저장한 리스트. 화상에 존재하는 물체 수만큼 요소를 가진다

        """

        # 화상 내 모든 물체의 어노테이션을 이 리스트에 저장
        ret= []

        # xml 파일 로드
        xml= ET.parse(xml_path).getroot()

        # 화상 내 물체(object) 수 만큼 반복
        for obj in xml.iter('object'):

            # 어노테이션에서 검지가 difficult로 설정된것은 제외
            difficult=int(obj.find('difficult').text)
            if difficult==1:
                continue

            # 한 물체의 어노테이션을 저장하는 리스트
            bndbox=[]

            name=obj.find('name').text.lower().strip()
            bbox=obj.find('bndbox')

            # 어노테이션의 xmin, ymin, xmax, ymax를 취득하고 0~1로 normalization
            pts=['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOC는 원점을 (1,1)로 하기에 1을 빼서 원점을 (0,0)으로 조정한다
                cur_pixel=int(bbox.find(pt).text)-1

                # 폭, 높이로 normalization
                if pt=='xmin' or pt=='xmax':
                    cur_pixel/=width
                else:
                    cur_pixel/=height
                bndbox.append(cur_pixel)

            # 어노테이션의 클래스명 index를 취득하여 추가

            label_idx=self.classes.index(name)
            bndbox.append(label_idx)

            ret+=[bndbox] # ret에 xmin, ymin, xamx, ymax, label_ind를 더한다
        return np.array(ret) #
```
출력은 `xmin,ymin,xmax,ymax,label_ind`형태이다
동작이 잘 되는지 확인해보자
```
# 동작 확인
voc_classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
transform=Anno_xml2list(voc_classes)

# 화상 로드용으로 OpenCv사용
ind=1
image_file_path=val_img_list[ind]
img=cv2.imread(image_file_path) # 화상을 읽어 들임
height, width, channels=img.shape

# 어노테이션을 리스트로 표시
transform_anno=transform(val_anno_list[ind], width, height)
print(transform_anno)
# [xmin, ymin, xmax, ymax, label_ind]
# 18 = train
# 14 = person
```
>[[ 0.09        0.03003003  0.998       0.996997   18.        ]
 [ 0.122       0.56756757  0.164       0.72672673 14.        ]]
 
잘 출력된다. train과 person객체가 있음을 알 수 있으며 BBox좌표도 알 수 있다

#### 💬 3.image와 annotation data를 전처리하는 클래스 작성
**image에 변형이 일어나면 BBOX도 같이 변형시켜줘야한다**
저자의 깃헙에 있는 .py 파일에서 augmentation 클래스를 사용한다
```
# utils 폴더에 있는 data_augmentation.py 가져와서 사용
# 입력 영상의 전처리 class 작성
from utils.data_augumentation import Compose, ConvertFromInts,ToAbsoluteCoords,PhotometricDistort,Expand,RandomSampleCrop,RandomMirror,ToPercentCoords,Resize,SubtractMeans

class DataTransform():
    """
    화상과 어노테이션의 전처리 클래스, 훈련과 추론에서 다르게 작성
    화상 크기는 300x300
    augmentation 수행

    Attributes
    ----------
    input_size: int
    color_mean: (B,G,R)
    """
    def __init__(self, input_size, color_mean):
        self.data_transform={
            'train':Compose([ConvertFromInts(), ToAbsoluteCoords(),PhotometricDistort(),Expand(color_mean),RandomSampleCrop(),RandomMirror(),ToPercentCoords(),Resize(input_size),SubtractMeans(color_mean)]),
            'val':Compose([ConvertFromInts(), ToAbsoluteCoords(),Resize(input_size),SubtractMeans(color_mean)])
        }

    def __call__(self,img,phase,boxes,labels):
        return self.data_transform[phase](img,boxes,labels)
```

#### 💬 4.DataSet 작성하기
- 전처리한 image의 tensor형식 data와 annotation을 얻어낸다
```
# VOC2012 Dataset 작성

class VOCDataset(data.Dataset):
    """
    VOC2012의 Dataset을 만드는 클래스

    Attributes
    -----
    img_list
    anno_list
    phase
    transform
    transform_anno
    """
    def __init__(self,img_list,anno_list,phase,transform,transform_anno):
        self.img_list=img_list
        self.anno_list=anno_list
        self.phase=phase
        self.transform=transform
        self.transform_anno=transform_anno

    def __len__(self):
        return len(self.img_list) # 화상 매수 반환
    
    def __getitem__(self,index):
        im,gt,he,w=self.pull_item(index) #전처리한 화상의 텐서 형식 데이터와 어노테이션 취득
        return im,gt
    
    def pull_item(self,index):
        '''전처리한 화상의 텐서 형식 데이터, 어노테이션, 높이, 폭 취득'''

        #1.이미지 읽기
        image_file_path=self.img_list[index]
        img=cv2.imread(image_file_path) # 화상을 읽어 들임
        height,width,channels=img.shape

        #2. xml형식의 어노테이션 정보를 리스트에 저장
        anno_file_path=self.anno_list[index]
        anno_list=self.transform_anno(anno_file_path,width,height)

        #3.전처리 실시
        img,boxes,labels=self.transform(img,self.phase,anno_list[:,:4],anno_list[:,4])

        #색상 채널의 순서가 BGR이므로 RGB로 변경
        #높이,폭,채널->채널,높이,폭 변경
        img=torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)

        #BBOX와 라벨을 세트로 한 np.array작성,gt=ground truth
        gt=np.hstack((boxes,np.expand_dims(labels,axis=1)))

        return img,gt,height,width
```
잘 작동하는지 확인해본다
```
#동작 확인
color_mean=(104,117,123)
input_size=300

train_dataset=VOCDataset(train_img_list,train_anno_list,phase="train",transform=DataTransform(input_size,color_mean),transform_anno=Anno_xml2list(voc_classes))
val_dataset=VOCDataset(val_img_list,val_anno_list,phase="val",transform=DataTransform(input_size,color_mean),transform_anno=Anno_xml2list(voc_classes))

#출력 예
val_dataset.__getitem__(1)
```
>(tensor([[[   0.9417,    6.1650,   11.1283,  ...,  -22.9083,  -13.2200,
             -9.4033],
          [   6.4367,    9.6600,   13.8283,  ...,  -21.4433,  -18.6500,
            -18.2033],
          [  10.8833,   13.5500,   16.7000,  ...,  -20.9917,  -24.5250,
            -25.1917],
          ...,
          [ -23.9500,  -14.9000,   -1.7583,  ..., -108.6083, -111.0000,
           -117.8083],
          [ -28.2817,  -20.1750,   -5.5633,  ..., -104.9933, -111.8350,
           -119.0000],
          [ -20.4767,  -21.0000,  -12.6333,  ..., -107.1683, -115.7800,
           -117.1100]],
 
         [[  25.9417,   30.1650,   35.1283,  ...,  -18.0767,  -14.7250,
            -11.8533],
          [  31.4367,   33.6600,   37.8283,  ...,  -13.5017,  -10.8250,
            -10.3783],
          [  35.7917,   37.5500,   40.7000,  ...,  -11.8417,  -13.0750,
            -14.0167],
          ...,
          [  -1.9500,    7.1000,   20.2417,  ..., -101.9083, -102.0000,
           -109.7167],
          [  -6.2817,    1.8250,   16.4367,  ..., -100.0517, -103.6700,
           -111.0000],
          [   1.5233,    1.0000,    9.3667,  ..., -102.5017, -107.7800,
           -109.1100]],
 
         [[  45.2750,   55.1650,   62.1283,  ...,   12.8500,   22.0550,
             27.8167],
          [  50.8800,   58.3300,   64.4983,  ...,   15.8350,   21.5150,
             22.7967],
          [  56.0667,   60.5500,   65.1500,  ...,   15.6417,   14.8250,
             14.7083],
          ...,
          [  36.7167,   43.1000,   56.2417,  ...,  -94.7583,  -96.0000,
           -101.9000],
          [  32.3850,   37.8250,   52.4367,  ...,  -92.1617,  -96.0000,
           -101.8867],
          [  40.1900,   37.0000,   45.3667,  ...,  -94.5017,  -99.7800,
            -99.1467]]]),
            array([[ 45.,  10., 499., 332.,  18.],[ 61., 189.,  82., 242.,  14.]]))
        
잘 출력이 된다
### ✏️ 2.3 DataLoader 구현
- data를 mini batch로 꺼내기 위한 DataLoader class를 구현한다
- image data마다 annotation 정보, gt(ground truth, image내 물체 수)가 다르기 때문에 주의해야한다.
-> collate_fn def를 별도로 만들어놔야 한다.
#### 💬 1.collate_fn 구현
```
def od_collate_fn(batch):
    """
    Dataset에서 꺼내는 어노테이션 데이터의 크기는 화상마다 다르다.
    ex)화상 내 물체가 2개 ->(2,5), 3개 ->(3,5)
    변화에 대응하는 DataLoader를 만들기 위해 collate_fn을 만든다.
    """
    targets=[]
    imgs=[]
    for sample in batch:
        imgs.append(sample[0]) #smaple0 = 이미지
        targets.append(torch.FloatTensor(sample[1])) #sample1= 어노테이션 gt

    #imgs는 미니 배치 크기의 리스트
    #리스트 요소는 torch.Size([3,300,300])
    #torch.Size([batch_num,3,300,300])으로 변환
    imgs=torch.stack(imgs,dim=0)

    #targets은 gt
    #리스트의 크기 = 미니배치 크기
    #targets 리스트 요소느 [n,5]
    #n은 화상마다 다르며 화상 속 물체 수
    #5는[xmin,ymin,xmax,ymax,calss_index]

    return imgs,targets
```
잘 load되는지 확인해보자
```
# 확인해보기

batch_size=4
train_dataloader=data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=od_collate_fn)
val_dataloader=data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=od_collate_fn)

#사전형 변수에 정리
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#동작확인
batch_iterator=iter(dataloaders_dict["val"])
images,targets=next(batch_iterator)
print(images.size())
print(len(targets))
print(targets[0].size())

#배치 4개, 3채널, 300x300이미지
#화상 내 물체 1개 확인
```
>torch.Size([4, 3, 300, 300])
4
torch.Size([1, 5])

마지막으로 총 data 수를 확인해보자
```
print(train_dataset.__len__())
print(val_dataset.__len__())
```
>5717
5823

### ✏️ 2.4 네트워크 모델 구현
< model >
- vgg
- extras
- loc
- conf

< sources >
- source1: vgg(conv4_3) -> L2Norm (512,38,38)(채널,크기,크기)
- source2: vgg output (19x19)
- source3: source2 -> extras(conv1_2) , 10x10
- source4: source2 -> extras(conv2_2), 5x5
- source5: source2 -> extras(conv3_2), 3x3
- source6: source2 -> extras(conv4_2), 1x1

- source6은 커다란 하나의 물체를 감지
- source1은 작은 물체 감지
-> source6이 더 많은 conv를 거쳤으므로 image 속 작은 물체를 감지하는 정밀도가 큰 물체의 감지보다 작다

이 후 source들을 loc network에 거치면 DBox의 offeset 정보를 추출하고
conf network에 거치면 클래스의 신뢰도를 추출한다.


#### 💬 VGG 모듈 구현
```
from math import sqrt
from itertools import product

import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# VGG모델 구현
def make_vgg():
    layers = []
    in_channels = 3  

    # 반복문으로 VGG구현
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            # ceil은 소수점을 올려 정수로
            # 디폴트는 소수점 버려 정수로
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


# 확인
vgg_test = make_vgg()
print(vgg_test)

```
#### 💬 extra 모듈 구현
```
# extra모델 구현
def make_extras():
    layers = []
    in_channels = 1024  # vgg output을 input으로 받을 때 채널 수

    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    #RELU를 순전파에서 준비하고 모듈안에서 x

    return nn.ModuleList(layers)



extras_test = make_extras()
print(extras_test)
```

#### 💬 loc및 conf 모듈 구현
```
# loc_layers=디폴트 박스의 오프셋 출력
# conf_layer=디폴트 박스의 클래스 신뢰도 출력


def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    # VGG 22층, conv4_3(source1)의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # VGG 최종층, (source2)의 합성곱 층
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source3)의 합성곱 층
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source4)의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    #extra(source5)의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source6)의 합성곱 층
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)



loc_test, conf_test = make_loc_conf()
print(loc_test)
print(conf_test)
```
#### 💬 L2Norm 구현
```
#L2 Norm 층 구현
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__() #부모클래스의 생성자 실행
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # weight 초깃값
        self.reset_parameters()  # 파라미터 초기화
        self.eps = 1e-10

    def reset_parameters(self):
        '''결합파라미터의 scale 크기 값으로 초기화 실행'''
        init.constant_(self.weight, self.scale)  #weight값이 모두 scale(=20)이 된다

    def forward(self, x):
        '''38x38의 특징량에 대해 512 채널에 걸쳐 제곱합의 루트 구했다.
        38x38개의 값을 사용하여 각 특징량을 정규화한 후 계수를 곱하여 계산하는 층'''

   
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out
```


#### 💬 DBOX 구현
- source1,5,6에는 4개의 DBOX, 2,3,4에는 6개의 DBOX 준비
- 각각의 DBOX 크기도 다양하게 구현
```
# 디폴트 박스 출력하는 클래스
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 초기설정
        self.image_size = cfg['input_size']  # 이미지 크기 300
        # [38, 19, …] 각 source의 feature map의 크기
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"])  # source = 6개
        self.steps = cfg['steps']  # [8, 16, …] DBox의 pixel 크기
        
        self.min_sizes = cfg['min_sizes']
        # [30, 60, …] 작은 정사각형의 DBOX 면적
        
        self.max_sizes = cfg['max_sizes']
        # [60, 111, …] 큰 정사각형의 DBOX 면적
        
        self.aspect_ratios = cfg['aspect_ratios']  # DBOX의 종횡비

    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  
                # feature map 화상 크기
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # 정규화
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 화면비 1의 작으 DBOX [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 화면비 1의 큰 DBox [cx,cy, width, height]
                # 'max_sizes': [60, 111, 162, 213, 264, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 그외 화면비의 defBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # DBox를 텐서로 전환
        output = torch.Tensor(mean).view(-1, 4)

        # DBOX크기 조정
        output.clamp_(max=1, min=0)

        return output

```
DBOX정보가 잘 나오는지 확인해보자
```
# 동작 확인
# SSD300 설정
ssd_cfg = {
    'num_classes': 21,  
    'input_size': 300,
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4], 
    'feature_maps': [38, 19, 10, 5, 3, 1],  
    'steps': [8, 16, 32, 64, 100, 300], 
    'min_sizes': [30, 60, 111, 162, 213, 264],  
    'max_sizes': [60, 111, 162, 213, 264, 315], 
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# DBox
dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()

# DBox 출력확인
pd.DataFrame(dbox_list.numpy())
```
>
0	1	2	3
0	0.013333	0.013333	0.100000	0.100000
1	0.013333	0.013333	0.141421	0.141421
2	0.013333	0.013333	0.141421	0.070711
3	0.013333	0.013333	0.070711	0.141421
4	0.040000	0.013333	0.100000	0.100000
...	...	...	...	...
8727	0.833333	0.833333	0.502046	1.000000
8728	0.500000	0.500000	0.880000	0.880000
8729	0.500000	0.500000	0.961249	0.961249
8730	0.500000	0.500000	1.000000	0.622254
8731	0.500000	0.500000	0.622254	1.000000
8732 rows × 4 columns

#### 💬 SSD class 구현
위에 모듈들을 다 세팅해놨으므로 SSD class를 구현해보자
```
# SSD 클래스 작성
class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train or inference
        self.num_classes = cfg["num_classes"]  # 클래스 수 =21

        # SSD 네트워크 작성
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox작성
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # inference시 Detect()
        if phase == 'inference':
            self.detect = Detect()


# 동작확인
ssd_test = SSD(phase="train", cfg=ssd_cfg)
print(ssd_test)
```
>SSD(
  (vgg): ModuleList(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
    (32): ReLU(inplace=True)
    (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    (34): ReLU(inplace=True)
  )
  (extras): ModuleList(
    (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (2): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (3): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (4): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (6): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  )
  (L2Norm): L2Norm()
  (loc): ModuleList(
    (0): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4-5): 2 x Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (conf): ModuleList(
    (0): Conv2d(512, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4-5): 2 x Conv2d(256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)

각각의 모듈들이 잘 구현됐다.
다음주에 순전파 함수부터 이어서 구현해보겠다.
