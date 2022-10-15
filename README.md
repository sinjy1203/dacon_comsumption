# 전력사용량 예측 AI 경진대회
>건물정보와 기후정보 데이터셋을 통해 전력사용량을 예측하는 대회

</br>

## 1. 제작 기간 & 참여 인원
- 2021년 05월 10일 ~ 2021년 06월 25일
- 개인으로 참여

</br>

## 2. 사용 기술
- python
- tensorflow
- RNN

</br>

## 3. file 설명
`main.py` training model & prediction
`dataset.py` sliding window, categorical & numerical data preprocessing 
`model.py` RNN model

</br>

## 4. 트러블 슈팅
### 데이터 전처리 문제
- 학습 데이터가 범주형과 숫자형으로 있어서 일괄적으로 전처리하기 힘들었다.
- 범주형 데이터와 숫자형 데이터를 따로 전처리 후 합치는 방식으로 하였다.
