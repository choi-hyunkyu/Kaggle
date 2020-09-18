import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from BinaryFunction import BinaryClassifier

PATH = './data/'
model =  torch.load(PATH + 'model.pt') # 모델 전체 불러오기, 클래스 선언 필수

train_data = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\train_data.csv')
target = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\target.csv')
test_data = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\test_data.csv')

# DataFrame 벡터 변환
test_data_tensor = []
for vector in test_data.values:
    test_data_tensor.append(list(vector))

x_test = test_data_tensor
x_test = torch.FloatTensor(x_test)
x_test.shape    

# test data 모델에 삽입
model_test = model(x_test)

# test data to numpy
model_test = model_test.detach().numpy()
model_test = model_test.tolist()

# test_data 결과를 0과 1로 치환하기 위해 2차원 리스트를 1차원 리스트로 변환
model_test = sum(model_test, [])

model_test_result = []
for i in model_test:
    if i > 0.5 :
        i = 1
        model_test_result.append(i)
    elif i <= 0.5 :
        i = 0
        model_test_result.append(i)
model_test_result

# submission dataframe을 만들기 위해 test.csv 불러온 후 dataframe으로 저장
test = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\test.csv')

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": model_test_result
    })

submission.to_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\submission_bc.csv', index=False)
