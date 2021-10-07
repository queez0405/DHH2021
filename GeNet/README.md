# GeNet
Digital Health Hackathon 2021 Medical AI Challenge </br>
https://www.digitalhealthhack.org/ai-1 </br>

## Operation
### Train
```
python main.py
```

### Test
```
python main.py --test=True
```

## 9월 15일 회의  
### 강준구
Deephit(https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16160) 기반으로 regression 모델 학습 예정
### 김동희  
tree 기반의 머신러닝 기법에서 gini index 사용하여 treat를 하지 않은 데이터와 treat를 거친 데이터에서 각각 feature의 importance를 추출하여 그 차이가 큰 feature를 고름. grid search를 통한 ML의 hyperparameter search와 importance 추출 과정을 코드 진행할 예정
### 김지희
MLP, SVM으로 treatment classification 실험, SVM에서 70%의 accuracy
### 이재윤
MLP base의 treatment classification 실험, 성능은 최고 54%, Survival Time regression 모델 학습 예정
### 임현택  
Self-attention에 기반한 regression 모델 학습, 이후 test data를 만들어 유전자 10개에 대한 실험 예정

## 9월 23일 회의  
### 강준구
Deephit regression 모델 학습 완료, clinical variable은 기존의 데이터를 사용, genetic을 하나씩 바꿔가며 예상 survival time 측정 예정
### 김동희  
Grid search를 통한 ML의 hyperparameter search와 importance 추출 진행
### 김지희
5개 머신러닝 기법으로 treat, non-treat의 loss 차이를 기반으로 feature importance 계산
### 이재윤
Naive Bayes 기법으로 후보군 탐색, regression으로 검증 예정
### 임현택  
Self-attention regression 모델로 다른 머신 러닝 모델에서 예측한 후보군 검증 예정
