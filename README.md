# FashionHow_AI
고객 대화 데이터에 맞는 패션 이미지 셋 추천을 위한 유사도 측정


### Code

main.py : 프로그램 실행을 위한 main 파일

data_io.py : 데이터 불러오기, 전처리, 저장

model.py : 이미지 임베딩, 대화 메모리, 유사도 측정을 위한 딥러닝 모델

mevac.py : train, test, predict

run_prel.sh : 사전 작업 실행

run_train.sh : 학습 실행

run_pred.sh : 예측값 파일 생성


### Data

ddata.txt : 고객과 추천 봇의 대화 데이터

mdata.txt : 각 이미지의 특성 메타 데이터

ac_eval_t1.dev : 고객과 추천 봇의 대화에 따른 이미지셋 순위 데이터


### Reference

데이터 출처 : 
- 논문 제목 : 인터랙션 기반 추천 시스템 개발을 위한 데이터셋 연구
- 저자 : 정의석, 김현우, 오효정, 송화전
- 발표처 : HCLT 2020 (2020 한글 및 한국어 정보처리 학술대회)

base embedding model : https://fasttext.cc/docs/en/crawl-vectors.html
