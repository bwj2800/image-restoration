MODEL_NAME = "resnet34_2"
MODEL_PATH = 'result/model/'+MODEL_NAME+'.h5'
DATA_PATH = 'datasets/training/source'
EPOCH = 100
BATCH_SIZE = 16

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from loader import CustomImageFolder

# GPU 사용 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 클래스 매핑 딕셔너리
class_mapping = {
    'bright_FF': 0,
    'bright_RF': 0,
    'dark_FF': 0,
    'dark_RF': 0,
    'bright_RL': 1,
    'bright_UD': 1,
    'dark_RL': 1,
    'dark_UD': 1,
}

# 데이터 전처리 및 로드
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def get_class_from_path(path):
    return class_mapping[os.path.basename(os.path.dirname(path))]


dataset = CustomImageFolder(root=DATA_PATH, transform=transform,
                            target_transform=lambda x: get_class_from_path(x))

dataset.classes = [0, 1]

# 데이터를 train과 validation으로 나누기
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(train_dataset.dataset.classes)
print("클래스 수:", num_classes)


# ResNet 모델 정의
resnet_model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
# 분류 문제이므로 마지막 Fully Connected 레이어 수정
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model = resnet_model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
#                                            milestones=[30, 80],
#                                            gamma=0.1)

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

# 학습
for epoch in range(EPOCH):
    train_loss = 0.0
    valid_loss = 0.0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=train_iter):
        # print(type(images))
        # print(labels)
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred = resnet_model(images)
        # print(pred)
        loss = criterion(pred.double(), labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    total_train_loss = train_loss / train_iter
    # scheduler.step()

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = resnet_model(images)
            loss = criterion(pred.double(), labels)
            valid_loss += loss.item()

    total_valid_loss = valid_loss / valid_iter
    
    print("EPOCH: %f  [train loss / %f] [valid loss / %f]" % (epoch, total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        print("model saved\n")
        torch.save(resnet_model.state_dict(), MODEL_PATH)
        best_loss = total_valid_loss

