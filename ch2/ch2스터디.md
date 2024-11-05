## 🔎 1.Image Classification & Transfer Learning(VGG)

### ✏️ 1.1 학습된 VGG모델 사용
- 전이학습과 파인튜닝을 구현해보자
#### 💬 1.1.1 학습된 VGG 불러오기
```
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torchvision
from torchvision import models, transforms

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
```
필요한 라이브러리들을 improt해준다
```
use_pretrained = True
net=models.vgg16(pretrained=use_pretrained)
net.eval() # 모델을 평가 상태로 변경, dropout이나 batch normalization을 평가 모드로 변경
```
학습된 VGG모델을 불러오고나서 평가모드로 전환 후 모델 내부를 출력하면 아래와 같다.
>VGG(
  (features): Sequential(
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
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
...
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

- in_features와 out_channels에 대한 이해가 필요하다
여기서 feature모듈을 통과한 이미지는 512채널, 7x7 size를 갖게 되므로 (512,7,7) in_features가 25088이 되는 것이다.
이후 전결합층을 지나 총 out_feature 1000개로 분류하는 모델임을 알 수 있다.

#### 💬 1.1.2 전처리 Class 작성
```
class BaseTransform():
    """
    Attributes
    ------------
    resize: int
        크기 변경 전의 화상크기
    mean: (R, G, B)
        각 색상 채널의 평균값
    std: (R, G, B)
        각 색상 채널의 표준편차
    """
    
    def __init__(self,resize,mean,std):
        self.base_transform=transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    
    def __call__(self,img):
        return self.base_transform(img)
```
- Compose를 거쳐 이미지를 전처리 시킨다
-> 이미지의 짧은 변을 `resize`로 설정
-> 이미지 중앙을 resize x resize 로 잘라내기
-> 이미지를 tensor로 변환
-> 이미지를 평균과 표준편차로 정규화
- __call__ 함수는 클래스를 함수처럼 호출 했을 때 호출되는 함수이다
```
# 화상 전처리 확인해보기

image_file_path = './data/goldenretriever-3724972_640.jpg'
img=Image.open(image_file_path)

plt.imshow(img)
plt.show()

resize=224
mean=(0.485,0.456,0.406)
std=(0.229,0.224,0.225)
transform=BaseTransform(resize,mean,std)
img_transformed=transform(img)

img_transformed=img_transformed.numpy().transpose((1,2,0)) # 텐서를 numpy로 변환하고 축을 변환(PIL 이미지는 heigh*width*channel, 텐서는 channel*height*width)
img_transformed=np.clip(img_transformed,0,1) # 이미지 픽셀 값을 0~1로 정규화

plt.imshow(img_transformed)
plt.show()

```
- 전처리된 이미지를 불러와보자
- 이때 이미지 텐서를 numpy 형식에 맞게 transpose 시키고 시각화 해본다

#### 💬 1.1.3 후처리 Class 생성
- 모델의 1000개의 class 출력을 라벨로 변환하는 클래스를 만들어보자
```
# 라벨 정보를 불러와서 사전형 변수 생성
ILSVRC_class_index=json.load(open('./data/imagenet_class_index.json','r'))
ILSVRC_class_index
```
- 라벨 정보는 해당 json파일에 담겨있다.
>{'0': ['n01440764', 'tench'],
 '1': ['n01443537', 'goldfish'],
 '2': ['n01484850', 'great_white_shark'],
 '3': ['n01491361', 'tiger_shark'],
 '4': ['n01494475', 'hammerhead'],
 '5': ['n01496331', 'electric_ray'],
 '6': ['n01498041', 'stingray'],


ㅣ
 
 ```
 # 예측 후 확률이 가장 높은 라벨을 예측 결과로 출력하는 함수
class ILSVRCPredictor():
    """
    ISLVRC데이터 모델의 출력에서 라벨을 구한다.

    Attributes
    ------------
    class_index: dictionary
        키는 라벨, 값은 라벨 이름
    """

    def __init__(self,class_index):
        self.class_inedx=class_index
    
    def predict_max(self,out):
        """
        확률이 가장 높은 ILSVRC 라벨을 예측한다.

        Parameters
        ------------
        out: torch.Size([1, 1000])
            Net의 출력

        Returns
        ------------
        predicted_label_name: str
            예측한 라벨 이름
        """
        maxid=np.argmax(out.detach().numpy()) #넘파이 변환을 위해 detach()사용 후 numpy 변환 -> 가장 높은 값의 인덱스 추출
        predicted_label_name=self.class_inedx[str(maxid)][1] 

        return predicted_label_name

```
- out은 tensor 값을 가지므로 detach()를 이용하여 네트워크에서 분리후 numpy로 변환시켜 numpy연산을 수행한다

#### 💬 1.1.4 모델 예측
```
ILSVRC_class_index=json.load(open('./data/imagenet_class_index.json','r'))
predictor=ILSVRCPredictor(ILSVRC_class_index) # predict하는 클래스 생성


imgage_file_path='./data/goldenretriever-3724972_640.jpg'
img=Image.open(image_file_path)

transform=BaseTransform(resize,mean,std)
img_transformed=transform(img) # 위에서 선언한 전처리 클래스로 전처리
inputs=img_transformed.unsqueeze_(0) # 배치 차원 추가 (배치 수 , 컬러, 높이 너비)

out=net(inputs)
result=predictor.predict_max(out) # 모델의 출력의 결과값을 예측하는 함수로 예측 결과 출력

print("입력 화상의 예측 결과: ",result)
```
>입력 화상의 예측 결과:  golden_retriever

- `img_transformed.unsqueeze_(0)`을 이용하여 이미지의 배치 차원을 추가시킨다
-> VGG 모델의 input값은 (배치 수, 채널, height, width)이므로 배치수 차원을 갖게끔 이미지를 차원을 확장시켰다.

- 이로써 VGG 모델을 불러와 간단한 이미지 분류를 해보았다.

### ✏️ 1.2 딥러닝  구현 흐름

#### 1. 전처리, 후처리, 네트워크 모델의 입출력 확인
-> 전처리를 어떻게 할 것인가, 후처리는 어떻게 할 것인가, 어떤 모델을 쓸 것이고 그 모델의 계층은 어떻게 구현되어 있는가
#### 2. 데이터셋 작성
-> train set, val_set에 대한 Dataset 작성, Dataset Calss 작성

#### 3. 데이터 로더 작성
-> data set을 어떻게 가져올 것인지 작성, DataLoader Class 작성

#### 4. 네트워크 모델 작성
-> 모델을 어떻게 가져올 것인가, 어떻게 튜닝할 것인가

#### 5. 순전파 정의
-> 2장에서 다룰 예정

#### 6. 손실함수 정의
-> 어떤 손실함수를 사용할 것인가

#### 7. 최적화 기법 설정
-> 어떤 Optimization을 사용할 것인가

#### 8. 학습/검증
-> train & valid로 모델 성능 확인

#### 9. 테스트
-> test set으로 모델 최종 성능 확인

### ✏️ 1.3 전이학습 구현
- 학습된 모델을 기반으로 `최종 출력층 부근`을 own data에 맞게 조절하여 학습하는 기법
- 입력층까지 다시 학습시키는 `파인튜닝` 기법은 다음 절에 알아본다
#### 1. 전처리, 후처리, 네트워크 모델의 입출력 확인
- VGG16을 가져오고, 전처리로 정규화 ,증강 사용, 후처리로 labeling 한다. 
- 필요한 lib들을 import해서 가져온다


```
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

os.getcwd()
```

#### 2. 데이터셋 작성
```
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        """ phase: 'train' or 'val' """
        return self.data_transform[phase](img)
```
- `훈련시`에는 data augmentation을 이용하여 data를 증강시킨다.
```
# 훈련 시 화상 전처리 동작 확인
# 실행할 때마다 처리 결과 화상이 바뀜을 알 수 있음

img_file_path='./data/goldenretriever-3724972_640.jpg'
img=Image.open(img_file_path) # h,w,c 순서
print(img.size)
plt.imshow(img)
plt.show() # 원본 파일 표시

size=224
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transform=ImageTransform(size, mean, std)
img_transformed=transform(img, phase="train") # 전처리된 이미지

img_transformed=img_transformed.numpy().transpose((1, 2, 0)) # c,h,w 순서로 변환
img_transformed=np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show() # 전처리된 파일 표시
```
- augmentation이 잘 이루어졌는지 확인해본다
- `transform=ImageTransform(size, mean, std)` 클래스를 transform 변수에 저장한다
-> 이후 이 변수를 함수처럼 호출하여 `transform(img, phase="train")` `__call__ ` 클래스 내부의 함수 return값`self.data_transform[phase](img)`를 실행시키고 이 코드는 다시 
`  self.data_transform `을 사용자가 입력한 `phase`에 맞춰 실행시킨다

```
# 개미와 벌이 담긴 화상 파일으 경로 리스트 작성

def make_datapath_list(phase="train"):
    """
    데이터의 경로를 지정한 리스트 작성

    Parameters
    ----------
    phase : 'train' or 'val'
        훈련 데이터 또는 검증 데이터를 지정

    Returns
    -------
    path_list : list
        데이터의 경로를 저장한 리스트
    """
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg') # 현재 폴더의 하위 폴더까지 검색하여 jpg 파일을 찾음
    print(target_path)

    path_list=[]

    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list


triand_list=make_datapath_list(phase="train")
val_list=make_datapath_list(phase="val")

triand_list
```
- 개미와 벌을 classify하기 위해 경로 리스트를 작성한다
>['./data/hymenoptera_data/train\\ants\\0013035.jpg',
 './data/hymenoptera_data/train\\ants\\1030023514_aad5c608f9.jpg',
 './data/hymenoptera_data/train\\ants\\1095476100_3906d8afde.jp
 

```
class HymenopteraDataset(data.Dataset):

    def __init__(self, file_list,transform=None,phase='train'):
        self.file_list=file_list
        self.transform=transform
        self.phase=phase
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,index):
        
        img_path=self.file_list[index]
        img=Image.open(img_path)

        img_transformed=self.transform(img,self.phase)

        if self.phase=='train':
            label=img_path[30:34] # 파일 경로에서 라벨을 추출
        elif self.phase=='val':
            label=img_path[28:32]
        
        if label=="ants":
            label=0
        elif label=="bees":
            label=1
        
        return img_transformed,label

train_dataset=HymenopteraDataset(file_list=triand_list,transform=ImageTransform(size,mean,std),phase='train')
val_dataset=HymenopteraDataset(file_list=val_list,transform=ImageTransform(size,mean,std),phase='val')

index=0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])
        
```
 - Dataset class를 작성하여 train data set가 valid data set을 만든다.
 - 이미지가 개미=0, 벌=1로 지정한다
 -> 이때 label은 파일 경로에서 라벨을 추출해낸다. ` label=img_path[30:34]`
 
#### 3. 데이터 로더 작성 
```
 batch_size=32

train_dataloader=torch.utils.data.DataLoader(
    train_dataset,batch_size=batch_size,shuffle=True
)

val_dataloader=torch.utils.data.DataLoader(
    val_dataset,batch_size=batch_size,shuffle=False
)

dataloaders_dict={
    'train':train_dataloader,
    'val':val_dataloader
}

batch_iterator=iter(dataloaders_dict['train'])
inputs,labels=next(batch_iterator)

print(inputs.size())
print(labels) # 32개의 라벨이 출력됨
```
- 다 만든 data set을 어떻게 가져올 것인지 선언한다.
- train data는 data를 골고루 가져오가 위해 shuffle해서 가져온다.
>torch.Size([32, 3, 224, 224])
tensor([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 0, 0])

- 배치32, 채널3(RGB), 224 x 224 사이즈의 이미지임을 알 수 있고 각각 어떤 라벨을 갖는지 알 수 있다.
#### 4. 네트워크 모델 작성

```
use_pretrained=True
net=models.vgg16(pretrained=use_pretrained)

print(net.classifier[6]) # Linear(in_features=4096, out_features=1000, bias=True) -> out feature를 2개로 변경해야함

net.classifier[6]=nn.Linear(in_features=4096,out_features=2)
print(net.classifier[6])

net.train()
print('훈련 시작')

```
- 학습된 VGG16을 가져온다.
- 이때 최종 출력층 `classifier[6]` 부분을 우리의 data에 맞게 out feature = 2로 수정한다(개미, 벌)
- 최종 출력층을 수정 후 학습 시킨다.
>
VGG16의 classifier 부분은 다음과 같은 구조이다.
classifier[0] : Linear(25088, 4096)
classifier[1] : ReLU()
classifier[2] : Dropout(p=0.5)
classifier[3] : Linear(4096, 4096)
classifier[4] : ReLU()
classifier[5] : Dropout(p=0.5)
classifier[6] : Linear(4096, 1000)

Convolutional layer 부분
Classifier 부분으로 구성

지금 우리가 하고 있는 것은 최종 마지막 출력층을 2개로 변경하고 출력층을 학습시키는 것
"""


#### 6. 손실함수 정의
```
 criterion=nn.CrossEntropyLoss() # 손실함수 설정
```
- 분류 문제이므로 Corss Entropy를 사용한다.

#### 7. 최적화 기법 설정
```
params_to_update=[]

update_param_names=["classifier.6.weight","classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad=True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad=False

print('-----------------')
print(params_to_update)
```
- classifier의 6번째 layer의 가중치와 편항을 학습시키기 위해 해당 부분만 `param.requires_grad=True`로 설정하고 나머지 부분은 freeze 시킨다.
```
optimizer=optim.SGD(params=params_to_update,lr=0.001,momentum=0.9)
```
- SGD를 사용한다.
#### 8. 학습/검증
```
def train_model(net,dataloaders_dict,criterion,optimizer,num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-----------------')

        for phase in ['train','val']:
            if phase=='train':
                net.train()
            else:
                net.eval()
            epoch_loss=0.0
            epoch_corrects=0

            if(epoch==0) and (phase=='train'):
                continue

            for inputs,labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs=net(inputs)
                    loss=criterion(outputs,labels)
                    _,preds=torch.max(outputs,1) # 최댓값과  인덱스를 반환 -> 최댓값은 사용하지 않음, 1은 행을 기준으로 최댓값 찾으라는 뜻

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss+=loss.item()*inputs.size(0)
                    epoch_corrects+=torch.sum(preds==labels.data)
        
        epoch_loss=epoch_loss/len(dataloaders_dict[phase].dataset)
        epoch_acc=epoch_corrects.double()/len(dataloaders_dict[phase].dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                

```
- train일 때 backward를 진행시키고 학습시킨다.
- eval일 때는 학습하지 않는다.
- `_,preds=torch.max(outputs,1)` dim=1 기준으로 최대 값을 뽑아내어 예측을 실시한다.
```
num_epochs=2
train_model(net,dataloaders_dict,criterion,optimizer,num_epochs)
```
- 에폭을 2로 설정하여 모델을 학습했다.
```
Epoch 1/2
-----------------
100%|██████████| 5/5 [00:41<00:00,  8.28s/it]
val Loss: 0.7191 Acc: 0.4967
Epoch 2/2
-----------------
100%|██████████| 8/8 [01:02<00:00,  7.86s/it]
100%|██████████| 5/5 [00:46<00:00,  9.29s/it]
val Loss: 0.1726 Acc: 0.9608
```

### ✏️ 1.4 파인튜닝 구현
- 1.3에서는 출력층만 학습시켰지만 이번엔 feature 계층과 다른 classify 계층들도 학습시켜보자

```
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from tqdm import tqdm

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
```
- 필요한 라이브러리들을 가져온다.
- seed를 고정시켜 이후에도 같은 값을 얻도록 조정한다.

#### 데이터 셋과 로더 작성
```
from utils.dataloader_image_classification import ImageTransform, make_datapath_list,HymenopteraDataset

train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

size=224
mean=(0.485,0.456,0.406)
std=(0.229,0.224,0.225)
train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size,mean,std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size,mean,std), phase='val')

bathc_size = 32
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bathc_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=bathc_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
```
- 1.3 절에서 설명했으므로 자세한 설명은 생략하겠다.
#### 네트워크 모델 작성, 손실함수 정의
```
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6]=nn.Linear(in_features=4096,out_features=2)

net.train()

print('네트워크 설정 완료')
```
```
criterion = nn.CrossEntropyLoss()
```
#### 최적화 방법 설정
```
params_to_update_1 = [] # CNN층
params_to_update_2 = [] # 출력층과 가까운 층
params_to_update_3 = [] # 출력층

update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1에 저장:", name)
    
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2에 저장:", name)
    
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3에 저장:", name)
    
    else:
        param.requires_grad = False
        print("경사 계산 없음:", name)
```
- 각 layer마다 구분하여 list에 담는다.
- 목표 layer를`param.requires_grad = True`로 설정하여 학습 준비를 한다.

```
optimizer=optim.SGD([
    {'params':params_to_update_1, 'lr':1e-4},
    {'params':params_to_update_2, 'lr':5e-4},
    {'params':params_to_update_3, 'lr':1e-3}
], momentum=0.9)
```
- optimzier로 SGD를 사용하고. 학습시키고자 하는 계층에 따라 다른 learning rate를 적용시킨다.

- 이로써 이전과 달리 출력층만 학습시키는 것이 아닌 모델의 모든 층을 학습시킨다.

#### 모델 학습 및 검증
```
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치: ", device)

    net.to(device)

    torch.backends.cudnn.benchmark = True # 계산이 일정하다면 고속화 진행

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss=0.0
            epoch_corrects=0

            if (epoch == 0) and (phase == 'train'):
                continue
            
            #미니배치를 꺼내 루프 실행
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1) # dim=1기준

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
```
- `torch.backends.cudnn.benchmark = True` 입력 크기가 일관적이고 GPU에서 학습할 때 학습을 고속화 시키기 위해 사용했다.
```
num_epochs=2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
```
- 2에폭만 학습시켰다.
#### 모델 save & load
```
# 파라미터 저장
save_path='./weights_fine_tuning.pth'
torch.save(net.state_dict(), save_path)
```
```
# 파라미터 로드
load_path='./weights_fine_tuning.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

# GPU에 저장된 가중치를 CPU에 로드하는 경우
load_weights = torch.load(load_path, map_location={'cuda:0':'cpu'})
net.load_state_dict(load_weights)
```
- 모델의 가중치 정보들을 딕셔너리 형식으로 저장하고 불러온다.
- 이로써 이미지를 classify하는 파이프라인을 구현해보았으며 전이학습과 파인튜닝을 구현해봤다.
- 추가적으로 알아본 결과 파인튜닝 과정에서 본문에서는 전체 계층을 학습시켰지만 일부 필요한 계층만 학습시키고 나머지는 freeze시켜서 새롭게 학습시키는 파인튜닝 기법도 있다.
즉, 상황에 맞게 필요한 계층을 학습시키는 능력이 필요하다.
