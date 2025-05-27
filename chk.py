import pickle

# with open('./model/trained_columns.pkl', 'rb') as f:
#     print(pickle.load(f))
from pytorch_tabnet.tab_model import TabNetRegressor

model = TabNetRegressor()
model.load_model('assets/tabnet_model.zip')
print("Expected input features by model:", model.network.initial_bn.num_features)