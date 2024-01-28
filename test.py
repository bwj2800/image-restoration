MODEL_NAME = "resnet34_2"
MODEL_PATH = 'result/model/'+MODEL_NAME+'.h5'
DATA_PATH = 'datasets/validation/source'
EPOCH = 100
BATCH_SIZE = 16
ACCURACY_RESULT=''

f_acc=open('result/accuracy.txt','a')

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def get_class_from_path(path):
    return class_mapping[os.path.basename(os.path.dirname(path))]

test_dataset = CustomImageFolder(root=DATA_PATH, transform=transform,
                                 target_transform=lambda x: get_class_from_path(x))

test_dataset.classes = [0, 1]
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(test_dataset.classes)
print("클래스 수:", num_classes)


# ResNet 모델 정의
resnet_model = models.resnet34(weights=None)  # 이전에 학습된 가중치 사용하지 않음
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load(MODEL_PATH))
resnet_model.to(device)
resnet_model.eval()

# # tensor image generate
# images = test_transformer(Image.open('test_img/cat2.jpg')).view(1, 3, 224, 224)
# images = images.to(device)
# pred = resnet_model(images)
# # pred_sigmoid = torch.sigmoid(pred)
# # print(pred_sigmoid)
# # pred_rounded = torch.round(pred_sigmoid)
# pred_rounded = torch.round(pred)
# tmp=pred_rounded.cpu().detach().numpy()[0]
# print(tmp)


# 정확도 측정을 위한 변수 초기화
correct = 0
total = 0

# 평가를 위해 그라디언트 계산 비활성화
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 정확도 출력
test_accuracy = (correct / total) * 100
print('Test Accuracy of the model on the test images: {} %'.format(test_accuracy))

# 결과를 파일에 쓰기
ACCURACY_RESULT = MODEL_NAME + ' : ' + str(test_accuracy) + '\n'
f_acc.write(ACCURACY_RESULT)
f_acc.close()