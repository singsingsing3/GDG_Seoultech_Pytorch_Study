# Ch.0 파이토치 기본개념 정리

## 🔎 Tensros
### ✏️ 텐서의 선언과 특징

- 파이토치에서 기본 구조로 텐서를 사용한다
- Numpy와 유사하지만 토치에서 텐서를 사용하는 이유는 텐서는 GPU를 이용한 연산 가속이 가능하기 때문이다
- 텐서와 넘파이 간의 변환도 가능하다

#### 💬 텐서 초기화
```
x=torch.rand(4,2)
print(x)
4행 2열짜리 텐서를 만들고 값을 랜덤으로 초기화 한다
```
```
x=torch.randn_like(x,dtype=torch.float)
print(x)
텐서 x와 같은 크기의 텐서를 만들고 내용 데이터 타입은 float으로 선언
이 때 값들은 정규분포를 갖는 분포에서 랜덤값들을 뽑아와서 채운다
```
```
device=torch.device('cuda' if torch.cuda.is_available() else ' cpu') # gpu가능하면 gpu사용 아니면 CPU사용
print(device)
y=torch.ones_like(x,device=device) # x와 shape가 같은 1로 채워진 텐서를 GPU 혹은 CPU에 저장
print(y)
x=x.to(device) # x값을 GPU or CPU에 넣기
print(x)
z=x+y
print(z)
print(z.to('cpu',torch.double)) #z값 cpu로 옮기고 double형으로 저장
print(z)
```
이 처럼 텐서는 Numpy와 유사하지만 CPU와 GPU를 넘나드며 연산을 수행할 수 있다
#### 💬 Numpy <-> Tensor
```
a=torch.ones(7)
print(a)
b=a.numpy()
print(b)
a.add_(1)
print(a)
print(b) # a값이 변했는데 b값도 변함
```
텐서 a를 1로채운 크기가 7인 텐서로 선언하고
b는 a텐서를 numpy로 바꿔서 선언한다

그다음 a텐서에 1을 더한 값을 a텐서에 저장하고(in-place방식, 아래에서 설명)

a,b를 출력한다

만약 텐서 a가 CPU에 있었다면 a와 b는 같은 메모리 공간을 공유하므로 a가 변하면 b도 변함을 알 수 있다.

텐서는 cpu와 gpu를 넘나드며 연산할 수 있고 numpy와 tensor의 형변환도 할 수있다


#### 💬 텐서의 최대 최소 위치 출력
```
x=torch.rand(2,2)
print(x)
print(x.min(dim=0)) #행 기준 최소값 위치 출력
print(x.min(dim=1)) # 열 기준 최소값 위치 출력
```
텐서가 아래와 같이 선언되었을 때
```
tensor([[0.2005, 0.4955],
        [0.9937, 0.4030]])      

```
```
print(x.min(dim=0)) #열 기준

```
dim=0으로 최소값 위치를 출력하면
```
indices=tensor([0, 1]))
```
dim=1로 최소값 위치를 출력하면
```
indices=tensor([0, 1]))
```
위처럼 출력된다

#### 💬 텐서의 in-place 방식
- in-place 방식에서는 연산뒤에 '_'가 붙는다
```
y.add_(x) # y에 x를 더한 값을 y에 저장
```
y텐서에 x텐서를 더하고 그 결과 값을 다시 y텐서에 저장한다는 뜻이다
마찬가지로 뺄셈, 곱셈, 나눗셈, 내적 연산도 가능하다

#### 💬 텐서의 size나 shape 변경
- view 함수를 사용하면된다
```
x=torch.randn(4,5)
print(x)
y=x.view(20)
print(y)
z=x.view(5,-1) # 행은 5개로 하되 열은 너가 알아서 맞춰라
print(z)
```
텐서 x를 4행 5열의 크기로 선언하고 값은 정규분포를 갖는 랜덤값들로 채운다

그후
```
y=x.view(20)
```
y라는 텐서에 x라는 텐서를 1*20으로 쭉 펼친 값으로 선언한다
여기서 view라는 함수가 텐서 x의 shape을 바꾸었다

```
z=x.view(5,-1)
```
z라는 텐서는 4X5크기의 x텐서를 5x?값으로 바꾸어서 선언한다
이때 ?는 컴퓨터가 알아서 연산한다
따라서 행은 5개로 하되 열의 크기는 컴퓨터에게 알아서 맡길 때 -1 값을 넣으면 된다

- view라는 함수를 사용하는 이유는 텐서의 크기나 모양을 바꾸기 위해서 사용된다
- 후에 CNN을 거쳐 FCN으로 텐서를 넘길 때 텐서를 채널 수를 맞추는 작업이 필요하다. 그 때에 view함수를 쓰게 된다


## 🔎 Autograd(자동미분)

- `torch.autograd` 패키지는 Tensor의 모든 연산에 대해 **자동 미분** 제공
- `backprop`를 위해 미분값을 자동으로 계산
- `requires_grad` 속성 True로 설정하면 해당 텐서에서 이루어지는 연산을 추적
- `.detach()`를 호출하면 연산기록으로부터 분리 시킨다

```
x=torch.ones(3,3, requires_grad=True)
print(x)

y=x+5
print(y) # add 연산 됨을 알 수있다

tensor([[6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.]], grad_fn=<AddBackward0>)
```
텐서 x를 3x3크기의 1로 채워진 텐서로 선언하고 연산을 추적한다

그 후 x에 5를 더한 값을 y 텐서로 선언하고 y를 출력한다
이 때 y의 grad_fn을 보면 AddBackward가 기록되었음을 알 수 있다.

즉 x텐서에 추적기를 달았고 y는 x로 이루어진 연산을 수행했다
그래서 y를 추적했을 때 add의 흔적이 남아있음을 알 수있다

이 특성은 나중에 backpropagation을 진행할 때 사용된다
```
y.backward()
```
함수를 이용하면 자동으로 역전파 계산을 수행하고
`.grad` 속성에 저장된다

추적을 중지시킬 때는 `.detach()` 함수를 사용할 수 있다
```
print(x.requires_grad)
y=x.detach()
print(y.requires_grad)

True
False
```
위에서 선언한 x는 연산을 추적하고 있어서 True로 출력되지만
y는 `.detach()`함수로 추적을 그만두었기에 False로 출력됨을 알 수 있다

## 🔎 데이터 다루기
```
from torch.utils.data import Dataset, DataLoader
```
데이터 준비를 위해 `torch.utils.data`로부터 `Dataset, DataLoader`를 불러온다

```
import torchvision.transforms as transforms
from torchvision import datasets
```
`torchvision`을 통해 datasets를 불러오고 전처리 메소드 `transforms`를 갖고온다

```
mnist_transfrom=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5),std=(1.0))])
```
전처리 구간에 들어갈 인자를 미리 선언하였다
`transforms.Compose`는 리스트 안에 있는 연산을 차례대로 수행한다
`trnasforms.ToTensor()`는 텐서로 형변환을 시킨다
`transforms.Normalize`는 설정한 평균과 표준편차로 데이터를 정규화 시킨다

즉 위의 코드는 데이터를 불러와서 데이터를 텐서 타입으로 바꾸고 정규화 시키는 작업을 `mnist_transform` 변수에 저장한 것이다

```
trainset=datasets.MNIST(root='/content/',
                       train=True,download=True,
                       transform=mnist_transfrom)
testset=datasets.MNIST(root='/content/',
                       train=False,download=True,
                       transform=mnist_transfrom)
```
`datasets.MNIST`를 이용하여 데이터를 불러오고 train set은 train=True로 갖고온다
이 때 데이터 전처리 작업은 위에서 선언한 `mnist_transform` 변수를 넣어 전처리 시킨다

```
train_loader=DataLoader(trainset,batch_size=8,shuffle=True)
test_loader=DataLoader(testset,batch_size=8,shuffle=False)
```
`DataLoader`를 통해 데이터를 갖고온다
위에서 선언한 trainset과 testset으로부터 배치 크기는8로 train은 섞어서 갖고오고 test는 test를 진행해야하기에 섞어서 가져오지 않았다


```
dataiter=iter(train_loader)
images,labels=next(dataiter)
images.shape, labels.shape
```
`iter()`를 이용하여 데이터를 순회하고 `next()`를 이용하여 순회하며 다음 값들을 갖고온다

```
(torch.Size([8, 1, 28, 28])
```
[배치크기,컬러채널,높이,넓이] 라는 뜻이다
즉 8개의 이미지를 갖고왔으며, 채널이 1이므로 흑백 이미지, 이미지 크기는 28*28이라는 뜻이다


## 🔎 신경망 구성
`torch.nn`패키지를 이용
```
import torch.nn as nn
```
### ✏️ nn.Linear 계층

```
input=torch.randn(128,20)
print(input)

m=nn.Linear(20,30) # 입력feature 20개 출력 feature는 30개
print(m)

output=m(input)
print(output)
print(output.size())
```
input으로 128*20의 크기를 가진 torch를 랜덤 정규분포로 선언하고
(128개의 sample들이 20개의 feature를 가지고 있음)
`nn.Linear(20,30)`으로 계층을 선언한다
이 때 선형계층은 20개의 feature를 받아내서 30개의 feature를 출력한다

input을 선형 계층에 넣고 출력값을 output변수에 저장한다
이 떄 output변수의 사이즈는 `120x30`이 되어있음을 알 수 있다



### ✏️ Convolution Layers

`nn.conv2d()`를 사용한다
```
nn.Conv2d(in_channels=1,out_channels=20,kernel_size=5,stride=1) # 입력 채널 개수 1, 출력 채널 개수 20, 커널사이즈, stride -> 순서대로 인자 알고 있기
```
각각의 파라미터는 순서대로 입력채널수, 출력채널수, 커널 크기, stride, padding, dilation이다.

```
layer=nn.Conv2d(1,20,5,1).to(torch.device('cpu'))
layer
```
layer계층은 Convolution layer로 선언하고 입력채널1, 출력채널 20, 커널은 5x5 stride는 1로 진행하는 계층으로 선언했다


```
input_data=torch.unsqueeze(images[0],dim=0)
print(input_data.size())

output_data=layer(input_data)
output=output_data.data
output_arr=output.numpy()
output_arr.shape # 통과후 모양
```
images는 8개의 batch를 가지므로 [8,1,28,28]를 갖는다
images[0]은 그중 첫번째 이미지이므로 [1,28,28]를 갖는다
`unsqueeze(images[0],dim=0)`을 이용하여 첫번째 이미지의 차원을 확장 시키고 그 기준은 `dim=0`으로 지정하여 [1,1,28,28]이 되도록 만든다

이제 입력채널이 1인 이미지 차원(배치크기,컬러채널,행,열)이 되었으므로 layer계층에 넣는다

출력된 값을 numpy로 바꾸고 shape를 찍어본다
이 때 numpy로 바꾼 이유는 후에 matplot을 이용하여 시각화 하기 위함인데 그 부분은 생략하겠다

layer에서 입력채널 1, 출력채널 20이므로 output_arr값은
[1,20,24,24]를 갖게된다

### ✏️ Pooling Layer
```
import torch.nn.functional as F
```
`torch.nn.functional`을 import한다
`torch.nn.MaxPool2d`도 많이 사용한다고 한다

```
pool=F.max_pool2d(output,2,2)
```
output 값을 max polling 한다. 이때 첫 '2'는 커널 사이즈를 두번째 '2'는 stride이다

### ✏️ 비선형 활성화

```
with torch.no_grad():
    flatten=input_image.view(1,28*28)
    lin=nn.Linear(784,10)(flatten)
    softmax=F.softmax(lin ,dim=1)
print(softmax)
```
단순 결과를 보기위해 기울기 계산없이 진행한다
flatten이라는 변수는 image를 Linear의 입력채널에 맞게 크기를  변환하고
Linear 모델에(입력784,출력10)넣는다 그리고 결과값을 lin변수에 넣는다

그다음 dim=1(열)축을 기준으로 lin결과값들의 feature들을 softmax함수에 넣어 결과값을 변수 `softmax`에 저장한다

마찬가지로 RELU도 가능하다



## 🔎 모델 정의
`nn.Module`를 이용한다

```
class Model(nn.Module):
    def __init__(self,inputs):
        super(Model, self).__init__()
        self.layer=nn.Linear(inputs,1)
        self.activation=nn.Sigmoid()

    def forward(self,x): # layer1개 통과후 활성화 함수
        x=self.layer(x)
        x=self.activation(x)
        return x
```

`nn.Moudule`를 상속받는 클래스를 생선한다

`__init__(self,inputs)`에서 필요한 변수들을 선언한다

`def forward(self,x)`에서
변수 x는 선형 레이어를 거친 후 sigmoid함수를 지나고 그 값을 return한다

코드를 조금 더 직관적으로 작성해보자

### ✏️ nn.Sequential
`nn.Sequential`안에 있는 모듈을 순차적으로 실행한다
__init__에서 네트워크 모델을 구현하고
forward()에서 코드를 가독성 높게 작성할 수 있다

```
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=30,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3=nn.Sequential(
            nn.Linear(in_features=30*5*5,out_features=10,bias=True),
            nn.ReLU(inplace=True),
        )

        def forward(self,x):
            x=self.layer1(x)
            x=self.layer2(x)

            x=x.view(x.shape(0),-1) # linear모델 들어가기 전에 쫙 피는 작업
            x=self.layer3(x)
            return x
```
layer1층~3층까지 `__init__`에작성하고 `forward`에서 각각의 layer를 불러오면 된다
이때 layer 3층은 linear 층이므로 위에서 언급한 `.view()`함수를 이용하여 Linear모델의 입력채널에 맞게 조정한다

### ✏️ Loss Fuction
* 파이토치의 주요 손실 함수
  - `torch.nn.BCELoss`: 이진 분류를 위해 사용
  - `torch.nn.CrossEntropyLoss`: 다중 클래스 분류를 위해 사용
  - `torch.nn.MSELoss`: 회귀 모델에서 사용
 
```
criterion=nn.MSELoss()
criterion=nn.CrossEntropyLoss()
```
### ✏️ Optimizer
- `zero_grad`()를 이용해 옵티마이저에 사용된 파라미터들의 기울기를 0으로 설정
- 파이토치의 주요 옵티마이저: `optim.Adadelta, optim.Adagrad, optim.Adam, optim.RMSprop, optim.SGD`
- optimizer는 `step()`을 통해 전달받은 파라미터를 모델 업데이트
- torch.optim.Optimizer(params, defaults)

## 🔎 모델 학습
```
class LinearRegressionModel(nn.Module):
    def  __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,x):
        pred=self.linear(x)
        return pred
```
간단하게 선형모델 class를 선언 후
```
model=LinearRegressionModel()
```
model 객체를 생성한다

```
import torch.optim as optim

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.001)
```
손실함수와 optimizer를 선언한다
이때 손실함수는 MSE, optim은 SGD를 사용한다
학습시킬 parameter는 model의 파라미터로 선언, 학습률은 0.001로 한다

```
epochs=100
losses=[]

for epoch in range(epochs):
    optimizer.zero_grad()


    y_pred=model(X)
    loss=criterion(y_pred,y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

```
에폭은 100
loss를 저장할 리스트 선언
학습 실행
`opti,izer.zero_grad()`: 반복문마다 기울기가 축적됨을 방지하기 위해 기울기 초기화

`y_pred`: model에 X값을 넣어 학습한 결과물
`loss`:손실함수를 기반으로 예측값과 실제값에 대한 loss 계산
`losses`: loss 저장
`loss.bacward()`: 모델에 가중치와 관련된 기울기 계산
`optimizer.step()`: 기울기를 기반으로 모델의 가중치 설정

기본적인 파이토치 개념과 함수들을 알아보았다
