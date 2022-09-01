# 자율주행 센서의 안테나 성능 예측 AI 경진대회 kops팀
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
## 1. 개요
- 자율주행 센서의 안테나 공정 데이터셋을 하이퍼파라미터 최적화 프레임워크인 Optuna를 활용하여 최적의 성능이 나오는 PCA를 탐색하여 데이터셋 전처리
- 전처리 된 데이터셋을 학습 데이터셋으로 넣어 Optuna를 활용하여 4개 머신러닝 모델을 하이퍼파라미터 튜닝
- 하이퍼파라미터 튜닝되지 않은 머신러닝 모델 13개와 하이퍼파라미터 튜닝된 머신러닝 모델 4개를 기반 모델로 하여 1차 예측 수행 후, 기반 모델의 예측 결과를 최종 데이터 세트로 하여 메타 모델로 최종 예측
- 개발 환경: Google Colab / GPU

## 2. 자율주행 센서의 안테나 공정 데이터셋
- DACON의 [자율주행 센서의 안테나 성능 예측 AI 경진대회](https://dacon.io/competitions/official/235927/data)의 정형 데이터셋
- 데이터 개수: 학습(Train) 데이터셋 (39607개), 테스트(Test) 데이터셋 (39608개)
- 메타 데이터: 비식별화된 X Feature에 대한 세부 설명 자료, 비식별화된 Y Feature에 대한 세부 설명 자료, 각 샘플의 정상, 불량을 판정할 수 있는 Y Feature 별 스펙 기준 자료

## 3. PCA 최적화 (건너뛰기 가능)
1. 220830_PCA_optimize_with_optuna.ipynb 파일 열기
2. 코드 실행환경 확인 (기본 실행환경 : Google Colab)
3. 1.1 데이터 입/출력 경로 지정에서 자신의 환경에 해당하는 환경 경로 지정
4. 1.2 필요 라이브러리 설치
5. 이후 코드 실행
6. 최적의 파라미터로 {'pca_1': 0, 'pca_2': 0, 'pca_3': 5, 'pca_4': 2, 'pca_5': 0, 'pca_6': 0, 'pca_7': 1, 'pca_8': 0, 'pca_9': 1, 'pca_10': 2} 선정

## 4. 하이퍼파라미터 튜닝 (건너뛰기 가능)
1. 220830_ML_hyperparameter_tuning_with_optuna.ipynb 파일 열기
2. 코드 실행환경 확인 (기본 실행환경 : Google Colab / GPU)
3. 1.1 데이터 입/출력 경로 지정에서 자신의 환경에 해당하는 환경 경로 지정
4. 1.2 필요 라이브러리 설치
5. 이후 코드 실행
6. 하이퍼파라미터 튜닝된 값은 data/tune_param 내 폴더에 *.pkl 파일로 저장

## 5. 스태킹 앙상블 모델
1. 220830_stacking_ensemble_modeling.ipynb 파일 열기
2. 코드 실행환경 확인 (기본 실행환경 : Google Colab / GPU)
3. 1.1 데이터 입/출력 경로 지정에서 자신의 환경에 해당하는 환경 경로 지정
4. 1.2 필요 라이브러리 설치
5. 4.3.2까지 코드 실행
6. 4.3.3 학습된 모델을 불러오거나 직접 학습에서 "# 학습된 모델 불러오기"를 통해 훈련된 모델을 불러오거나 "# 학습된 모델을 불러오지 않고 직접 학습하기"를 통해 모델에 대한 학습 가능
7. 학습 완료된 모델은 data/ML_model 내 *.pkl 파일로 저장 되어있음
7. 5 성능 평가를 통해 학습된 기반 모델 및 메타 모델의 성능 평가
8. 6 테스트 데이터 예측에서 테스트 데이터를 통해 최종 예측하며 submission.csv 파일로 저장

## 6. 성능 평가
- 예측 성능 (Cross-Validation)

| 모델                               | 스코어      |
| ---------------------------------- | ----------- |
| LinearRegression                   | 1.983296472 |
| Ridge                              | 1.983645821 |
| Lasso                              | 2.011257877 |
| ElasticNet                         | 2.011188457 |
| LassoLars                          | 2.012444171 |
| OrthogonalMatchingPursuit          | 1.99540754  |
| BayesianRidge                      | 1.983660117 |
| ARDRegression                      | 1.986684399 |
| GradientBoostingRegressor          | 1.961697557 |
| HistGradientBoostingRegressor      | 1.950305362 |
| XGBRegressor                       | 1.994067162 |
| LGBMRegressor                      | 1.948753644 |
| CatBoostRegressor                  | 1.946809224 |
| HistGradientBoostingRegressor_tune | 1.945230369 |
| XGBRegressor_tune                  | 1.9359791   |
| LGBMRegressor_tune                 | 1.93851026  |
| CatBoostRegressor_tune             | 1.938609746 |
| 앙상블 모델                        | 1.932119124 |
| LB public score                    | 1.934038836 |
| LB private score                   | 1.949861201 |

- 예측 소요 시간 

|                | CPU times | sys    | total    | Wall time |
| -------------- | --------- | ------ | -------- | --------- |
| 1회 시행       | 2min 44s  | 3.66 s | 2min 47s | 2min 31s  |
| 2회 시행       | 2min 34s  | 2.08 s | 2min 37s | 2min 19s  |
| 3회 시행       | 2min 38s  | 2.18 s | 2min 40s | 2min 22s  |
| 4회 시행       | 2min 37s  | 2.13 s | 2min 39s | 2min 22s  |
| 5회 시행       | 2min 38s  | 2.13 s | 2min 40s | 2min 21s  |
| 6회 시행       | 2min 39s  | 2.15 s | 2min 41s | 2min 23s  |
| 7회 시행       | 2min 35s  | 2.14 s | 2min 38s | 2min 19s  |
| 8회 시행       | 2min 35s  | 2.1 s  | 2min 37s | 2min 18s  |
| 9회 시행       | 2min 34s  | 2.11 s | 2min 36s | 2min 18s  |
| 10회 시행      | 2min 32s  | 3.48 s | 2min 35s | 2min 18s  |
| 평균 소요 시간 | 2min 36s  | 2.42s  | 2min 39s | 2min 21s  |

## 6. 전체 파일구조
```
./dacon-235927-kops
├── 220830_ML_hyperparameter_tuning_with_optuna.ipynb
├── 220830_PCA_optimize_with_optuna.ipynb
├── 220830_stacking_ensemble_modeling.ipynb
├── data
│   ├── meta
│   │   ├── x_feature_info.csv
│   │   ├── y_feature_info.csv
│   │   └── y_feature_spec_info.csv
│   ├── ML_model
│   │   ├── saved_ARDRegression.pkl
│   │   ├── saved_BayesianRidge.pkl
│   │   ├── saved_CatBoostRegressor.pkl
│   │   ├── saved_CatBoostRegressor_tune.pkl
│   │   ├── saved_ElasticNet.pkl
│   │   ├── saved_GradientBoostingRegressor.pkl
│   │   ├── saved_HistGradientBoostingRegressor.pkl
│   │   ├── saved_HistGradientBoostingRegressor_tune.pkl
│   │   ├── saved_LassoLars.pkl
│   │   ├── saved_Lasso.pkl
│   │   ├── saved_LGBMRegressor.pkl
│   │   ├── saved_LGBMRegressor_tune.pkl
│   │   ├── saved_LinearRegression.pkl
│   │   ├── saved_Meta_LinearRegression.pkl
│   │   ├── saved_Meta_Ridge.pkl
│   │   ├── saved_OrthogonalMatchingPursuit.pkl
│   │   ├── saved_Ridge.pkl
│   │   ├── saved_XGBRegressor.pkl
│   │   └── saved_XGBRegressor_tune.pkl
│   ├── sample_submission.csv
│   ├── submission.csv
│   ├── test.csv
│   ├── train.csv
│   └── tune_param
│       ├── CatBoostRegressor_tune
│       │   ├── tune_0.pkl
│       │   ├── tune_10.pkl
│       │   ├── tune_11.pkl
│       │   ├── tune_12.pkl
│       │   ├── tune_13.pkl
│       │   ├── tune_1.pkl
│       │   ├── tune_2.pkl
│       │   ├── tune_3.pkl
│       │   ├── tune_4.pkl
│       │   ├── tune_5.pkl
│       │   ├── tune_6.pkl
│       │   ├── tune_7.pkl
│       │   ├── tune_8.pkl
│       │   └── tune_9.pkl
│       ├── HistGradientBoostingRegressor_tune
│       │   ├── tune_0.pkl
│       │   ├── tune_10.pkl
│       │   ├── tune_11.pkl
│       │   ├── tune_12.pkl
│       │   ├── tune_13.pkl
│       │   ├── tune_1.pkl
│       │   ├── tune_2.pkl
│       │   ├── tune_3.pkl
│       │   ├── tune_4.pkl
│       │   ├── tune_5.pkl
│       │   ├── tune_6.pkl
│       │   ├── tune_7.pkl
│       │   ├── tune_8.pkl
│       │   └── tune_9.pkl
│       ├── LGBMRegressor_tune
│       │   ├── tune_0.pkl
│       │   ├── tune_10.pkl
│       │   ├── tune_11.pkl
│       │   ├── tune_12.pkl
│       │   ├── tune_13.pkl
│       │   ├── tune_1.pkl
│       │   ├── tune_2.pkl
│       │   ├── tune_3.pkl
│       │   ├── tune_4.pkl
│       │   ├── tune_5.pkl
│       │   ├── tune_6.pkl
│       │   ├── tune_7.pkl
│       │   ├── tune_8.pkl
│       │   └── tune_9.pkl
│       └── XGBRegressor_tune
│           ├── tune_0.pkl
│           ├── tune_10.pkl
│           ├── tune_11.pkl
│           ├── tune_12.pkl
│           ├── tune_13.pkl
│           ├── tune_1.pkl
│           ├── tune_2.pkl
│           ├── tune_3.pkl
│           ├── tune_4.pkl
│           ├── tune_5.pkl
│           ├── tune_6.pkl
│           ├── tune_7.pkl
│           ├── tune_8.pkl
│           └── tune_9.pkl
└── README.md
```
