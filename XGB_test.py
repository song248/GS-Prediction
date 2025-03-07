import xgboost as xgb
import numpy as np

loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_model.json")

print("모델 구조 확인:", loaded_model)
print("Feature 중요도:", loaded_model.feature_importances_)

X_dummy = np.random.rand(1, loaded_model.n_features_in_)
result = loaded_model.predict(X_dummy)

print("더미 데이터 예측 결과:", result)
