## 🔎 개요
- pytorch를 응용하여 VGG 모델을 구축하고 MNIST 손글씨를 classification 해보자.

### ✏️ 코드 설명
```
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```
우선 필요한 라이브러리들을 import해온다

- torch: 필수로 가져온다
- DataLoader: dataset으로부터 data를 load할 때 사용한다
- transforms: data를 증강하거나 정규화 등 전처리 작업에 사용한다
- datasets: MNIST 손글씨 data를 가져올 때 사용한다
- nn: 신경망 구축할 때 사용한다
- optim: optimizer를 설정할 때 사용한다
- plt: 데이터 시각화에 사용한다

```
mnist_transfrom=transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
- `transform`을 이용하여 data를 불러들일 때 해야할 작업들을 선언한다
- `Compose([])`: [] 안에 있는 작업들을 순차적으로 진행한다

- `transforms.Resize((224, 224))`이미지를 224*224 size로 조정한다
-> VGG NET의 input image size가 224이기에 조정했다
- `transforms.Grayscale(num_output_channels=3)` iamge를 Grayscale로 전환하지만 channel은 3으로 만든다
-> MNIST 손글씨는 주로 Grayscale로 작업하기에 Grayscale로 전환해 보았다. 그러나 VGG NET의 입력 채널 값은 3이기에 channel은 3으로 다시 만들었다. VGG NET의 입력 채널 값을 1로 바꾸어도 된다.
최대한 원본 그대로를 유지하고자 image의 channel을 바꾸었다
(Gray image일 경우 image channel= 1)
- `transforms.ToTensor()` 마지막으로 위의 과정들을 거친 이미지들을 tensor로 변환한다.
- `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` 각각 순서대로 설정한 mean과 std 값으로 정규화 시킨다.
이 값들은 상황에 맞게 설정하면 된다.
```
trainset=datasets.MNIST(root='/content/',
                       train=True,download=True,
                       transform=mnist_transfrom)
testset=datasets.MNIST(root='/content/',
                       train=False,download=True,
                       transform=mnist_transfrom)
```
- `datasets.MNIST` datasets 중 MNIST data들을 가져온다
이 때 파일 경로를 지정하고, trainset은 train=True로, testset은 train=False로 가져온다.
```
train_loader=DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
test_loader=DataLoader(testset,batch_size=32,shuffle=False,num_workers=2) 
```
- `DataLoader`를 이용하여 위에서 설정한 trainset을 가져온다
- batch size는 32로 설정했다. 
- train은 다양하게 훈련하기 위해 shuffle=True로 test는 검증해야하므로 shuffle해서 가져오지 않는다.
- num_workers는 데이터를 load할 때 사용할 프로세스 수이다.
빠르게 가져오기 위해 2로 설정했다.
```
dataiter=iter(train_loader)
images,labels=next(dataiter)
images.shape, labels.shape
```
- `iter()`를 이용하여 train_loader에 있는 train set들을 순환한다.
- `next()`를 이용하여 다음 순환으로 넘어간다
이 때 호출되는 image와 label의 shape를 출력해보자

>(torch.Size([32, 3, 224, 224]), torch.Size([32]))

image의 shape은 배치수32, channel=3, 224*224 사이즈이다.
label은 이미지 1장당 1개이므로 배치 수에 맞게 32개의 벡터로 구성되어있다.
```
img,label=trainset[4]
print(img.shape)
img_gray = img.mean(dim=0)
plt.title(label)
plt.axis('off')
plt.imshow(img_gray, cmap='gray')
plt.show()
```
data가 잘 들어왔는지 확인하기 위해 train set의 5번째 이미지를 가져와서 데이터를 시각화 해본다.

```
class VGG(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```
VGG 모델을 구현했다.
- `nn.Sequential` 함수는 ()안에 있는 연산들을 순차적으로 진행한다.
VGG의 구조를 설명하는 것보다 우선 pytorch 구현에 초점을 두겠다.
- ` nn.Linear(7 * 7 * 512, 4096)` classify하는 fc계층에서 첫 Linear계층은 7*7*512의 input 값을 갖는다.

- feature layer를 지나면 224x224의 image의 크기가 7x7로 변하고 channel은 512를 가진다.

- `x = x.view(x.size(0), -1)`를 이용하여 fc계층에 들어가기 전 1차원 벡터로 변환한다.
- 따라서 fc layer에 들어가는 값은 7x7x512값이 되는 것이다.
VGG NET을 구현할 때 공부하기위해 channel수들을 하드코딩하였다.
```
model = VGG(in_channels=3)
model
```
model를 설정하였고 input 이미지의 channel은 3이될 것으로 설정했다.
그리고 모델이 어떻게 구성되어있는지 출력했다.

```
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
```

- `CrossEntropyLoss()` 손실함수는 multi classfier를 위해 CrossEntropy를 사용하였다.
- `Adam` optimizer는 Adam을 사용하였고 learning late를 0.001로 설정하였다.
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
GPU를 사용할 수 있으면 사용하기 위해 위의 코드를 작성하였고
- `model.to(device)`GPU를 사용하면 model을 GPU로 처리한다
```
num_epochs = 1
losses = []
model.train()  # Ensure the model is in training mode

for epoch in range(num_epochs):
    running_loss = 0.0  # Initialize running loss for the epoch
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear the gradients

        # Move data to the device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss for the current batch
        losses.append(loss.item())
        running_loss += loss.item()

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}')
```
이제 모델을 학습하는 부분이다.
이 부분은 GPT의 도움을 받았다.
여기에 있는 코드는 이전 pytorch 기본 문법 기록편에 적었으니 설명은 생략하겠다.
>Epoch [1/1], Step [1500/1875], Loss: 2.2983
Epoch [1/1], Step [1600/1875], Loss: 2.2833
Epoch [1/1], Step [1700/1875], Loss: 2.3087
Epoch [1/1], Step [1800/1875], Loss: 2.3225
Epoch [1/1] Average Loss: 2.3022

최종 LOSS는 위와같이 나왔다.
```
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%')
```
위의 코드를 이용하여 정확도를 확인해본 결과다.
> Accuracy: 11.35%

성능이 매우 좋지 않다. 이는 아마도 단순한 이미지에 맞지않는 너무 깊은 계층을 갖는 VGG model을 사용해서 오히려 제대로 학습이 안된 것같다.

처음부터 끝까지 pytorch를 이용하여 data를 불러오고 VGG모델을 구축하고 모델을 train해보았다.

