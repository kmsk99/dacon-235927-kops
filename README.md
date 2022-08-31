# 자율주행 센서의 안테나 성능 예측 AI 경진대회 kops팀
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)
## 1. 개요

## 2. 자율주행 센서의 안테나 공정 데이터셋

## 3. PCA 최적화

## 4. 하이퍼파라미터 튜닝

## 5. 스태킹 앙상블 모델

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
├── LICENSE
└── README.md
```

## 📝 License
```
MIT License

Copyright (c) 2022 kmsk99

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
