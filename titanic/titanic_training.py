import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from BinaryFunction import BinaryClassifier

train_data = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\train_data.csv')
target = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\target.csv')

torch.manual_seed(1)

# DataFrame 벡터 변환
train_data_tensor = []
for vector in train_data.values:
    train_data_tensor.append(list(vector))

x_train = train_data_tensor
x_train = torch.FloatTensor(x_train)

target_data_tensor = []
for vector in target.values:
    target_data_tensor.append(list(vector))
    
y_train = target_data_tensor
y_train = torch.FloatTensor(y_train)


model = BinaryClassifier()

print(list(model.parameters())) # Weight와 Bias 확인

# optimizer 설정
learning_rate = 0.05
nb_epochs = 1000
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# 학습 진행

for epoch in range(nb_epochs + 1):

    # Hypothesis 계산
    hypothesis = model(x_train)

    # Cost 계산
    criterion = F.binary_cross_entropy(hypothesis, y_train)

    # Cost로 Hypothesis 개선
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()

    # 20번 마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, criterion.item(), accuracy * 100,
        ))
model(x_train)

# 모델 저장
PATH = './data/'
torch.save(model, PATH + 'model.pt') # 모델 전체 저장
