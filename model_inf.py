import os
import cv2
import numpy as np
import torch
import pickle
import time
import pandas as pd
from torch.autograd import Variable
from pytorch_dcsaunet.DCSAU_Net import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PyQt5.QtGui import QPixmap, QImage

class ImageModel:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, file_path):
        self.original_image = cv2.imread(file_path)

    def resize_image(self, width=300, height=300):
        if self.original_image is not None:
            self.processed_image = cv2.resize(self.original_image, (width, height))

    def convert_cv_qt(self, cv_img):
        if cv_img is None:
            return QPixmap()
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# def inference_single_image(image: np.ndarray) -> np.ndarray:
#     model_path = 'assets/epoch_last.pth'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     if len(image.shape) == 2:
#         image = cv2.merge([image, image, image])
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     transform = get_transform()
#     transformed = transform(image=image_rgb)
#     image_tensor = transformed['image'].unsqueeze(0).to(device)

#     model = torch.load(model_path, map_location=device)
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         output = model(image_tensor)
#         output = torch.sigmoid(output)
#         output = (output >= 0.5).float()

#     mask = output.squeeze().cpu().numpy()
#     mask = (mask * 255).astype(np.uint8)

#     print("Inference done!")
#     cv2.imwrite('output_mask.png', mask)
#     return mask
def inference_single_image(image: np.ndarray) -> np.ndarray:
    model_path = 'assets/epoch_last.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(image.shape) == 2:
        image = cv2.merge([image, image, image])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    start_time = time.time()  # 시간 측정 시작

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output >= 0.5).float()

    elapsed_time = time.time() - start_time  # 시간 측정 종료

    mask = output.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    print("Inference done!")
    print(f"Inference time: {elapsed_time:.4f} seconds")  # 추론 시간 출력

    cv2.imwrite('output_mask.png', mask)
    return mask
def predict_with_rfr(stats: dict) -> float:
    """
    stats: {
        'Min': float, 'Max': float, ..., 'Category': str, 'Steel': str, 'Lot': str
    }
    """

    # 1. 앙상블 모델 불러오기
    with open('./model/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 2. 전처리기 및 인코더 불러오기
    with open('./model/min_scaler.pkl', 'rb') as f:
        min_scaler = pickle.load(f)
    with open('./model/max_scaler.pkl', 'rb') as f:
        max_scaler = pickle.load(f)
    with open('./model/mean_scaler.pkl', 'rb') as f:
        mean_scaler = pickle.load(f)
    with open('./model/count_scaler.pkl', 'rb') as f:
        count_scaler = pickle.load(f)
    with open('./model/std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    with open('./model/median_scaler.pkl', 'rb') as f:
        median_scaler = pickle.load(f)

    with open('./model/div_encoder.pkl', 'rb') as f:
        category_encoder = pickle.load(f)
    with open('./model/steel_encoder.pkl', 'rb') as f:
        steel_encoder = pickle.load(f)
    with open('./model/lot_encoder.pkl', 'rb') as f:
        lot_encoder = pickle.load(f)

    # 유효값 검사
    valid_steels = ['STS304CUS4', 'STS304CUMS4', 'S304THES3', 'S304J3S2', '304LDS1',
       'S302', 'S302S1', 'S304HCMS7', 'S304HCS5', 'STS304J3S2',
       'STS304HCS5', 'STS304CUS', 'STS304S1', 'STS302S1', '301LDS1',
       '304LDSX', 'STS304CUMS2', '304LDS3', '304LDSI', '304LDSL',
       'STS304SI', 'STS304HS2', 'STS304HIS5', 'STS304SA', '316LDS1',
       'STS304SX', 'STS316VARS2', 'STS316VAR', 'STS316LS1', 'STS310SS2',
       'STS316LJSE', '304LDSB', 'STS304J3S', 'STS304HCMS7', 'STS304H1S5',
       'STS316LEPSB', '316LDSB', 'API316LDS1', '316LDSY', 'other']
    valid_lots = ['B3B0544800', 'B410066807', 'B320715101', 'B3C0494500', 'B3C0494401', 'B3B0451009', 'B3C0491901', 'B3B0545501', 'B3C0477900', 'B410101403', 'B410465200',
       'B410101402', 'B420026100', 'B420025700', 'B3C0155105', 'B410102100', 'B410101400', 'B410067200', 'B410101401', 'B3B0451010', 'B3B0451303', 'B420026700', 'B410464900',
       'B410066601', 'B410465500', 'B410466100', 'B3B0545302', 'B410065907', 'B410161000', 'B410161105', 'B410161501', 'B410161601', 'B3B0542100', 'B3B0542200', 'B3B0542400',
       'B410065902', 'B3B0451400', 'B390610500', 'B420043100', 'B410103300', 'B410159901', 'B3C0483502', 'B410161202', 'B3C0154200', 'B3C0155100', 'B3B0557100', 'B3B0557005',
       'B410129601', 'B3B0557000', 'B3B0556901', 'B3A0443300', 'B420042400', 'B3B0556600', 'B410069400', 'B340219702', 'B410101700', 'B410130000', 'B410131200', 'B3B0537500',
       'B410069000', 'B410068800', 'B370095700', 'B3A0443400', 'B410102401', 'B410120100', 'B410068707', 'B420153100', 'B410144600', 'B420025400', 'B3C0484200', 'B420027000',
       'B3C0376100', 'B3A0374000', 'B410131600', 'B3B0040701', 'B410130200', 'B410142300', 'B290728600', 'B3C0512607', 'B410541308', 'B410541309', 'B410541801', 'B3B0040801',
       'B420081800', 'B420087300', 'B420179503', 'B420092100', 'B3C0474101', 'B410406400', 'B3C0150100', 'B3C0150200', 'B3C0162500', 'B3C0474405', 'B410145100', 'B410547600',
       'B410312901', 'B410102400', 'B3B0041000', 'B410111000', 'B420042501', 'B3C0153901', 'B410144900', 'B390157300', 'B410131400', 'B410131000', 'B410135200', 'B410143900',
       'B3B0451007', 'B410130100', 'B410144000', 'B410154600', 'B410395000', 'B410526700', 'B410479700', 'B410536900', 'B3B0040600', 'B390616200', 'B3B0451002', 'B3B0550000',
       'B410103001', 'B410515100', 'B410547002', 'B3B0551500', 'B410514800', 'B3C0074900', 'B3C0153900', 'B410479800', 'B3C0382200', 'B380297302', 'B3C0155101', 'B410103000',
       'B3B0490201', 'B410112000', 'B410118500', 'B410119800', 'B410137100', 'B410604800', 'B410130700', 'B410160600', 'B410119900', 'B350603302', 'B3B0507505', 'B3B0507407',
       'B380392201', 'B410396200', 'B420168100', 'B420405200', 'B3C0399100', 'B390619500', 'B3C0486700', 'B410142800', 'B410542400', 'B410192300', 'B410192800', 'B410604700',
       'B360628801', 'B410586500', 'B420153300', 'B420153600', 'B410150300', 'B420152400', 'B410126500', 'B410151200', 'B410385600', 'B410573003', 'B320576001', 'B420177001',
       'B420092400', 'B3B0551600', 'B3B0490301', 'B430141600', 'B420407500', 'B410143400', 'B420214802', 'B420516200', 'B410158700', 'B420087400', 'B420177000', 'B3C0476400',
       'B410143300', 'B410598300', 'B410612300', 'B430053800', 'B420071000', 'B430305601', 'B410155200', 'B410576900', 'B3B0399302', 'B410596500', 'B410144700', 'B410397300',
       'B430241806', 'B410597601', 'B420085100', 'B420171800', 'B420084100', 'B3C0381600', 'B420092600', 'B3B0490100', 'B3B0490300', 'B430149400', 'B420407900', 'B370335907',
       'B3B0490401', 'B420083200', 'B430141400', 'B430121501', 'B410147700', 'B430148800', 'B420084200', 'B420084500', 'B3C0364500', 'B420084400', 'B420089000', 'B430109300',
       'B330852300', 'B420177500', 'B420177501', 'B420405300', 'B410155900', 'B420093500', 'B410155800', 'B420106400', 'B3B0212000', 'B3C0399110', 'B420162400', 'B420171300',
       'B430296300', 'B080478500', 'B370097700', 'B410150200', 'B3A0362709', 'B410315100', 'B3B0451401', 'B420048900', 'B410538300', 'B420405800', 'B3C0364100', 'B3C0135800',
       'B3C0138900', 'B3C0153800', 'B3C0154000', 'B3C0305900', 'B3C0476500', 'B3C0477000', 'B410102600', 'B410102601', 'B420262600', 'B410139200', 'other']

    steel = stats["Steel"] if stats["Steel"] in valid_steels else 'other'
    lot = stats["Lot"] if stats["Lot"] in valid_lots else 'other'

    # 3. 전처리 적용
    input_data = pd.DataFrame({
        'min_con_pt': [min_scaler.transform([[stats['Min']]])[0, 0]],
        'max_con_pt': [max_scaler.transform([[stats['Max']]])[0, 0]],
        'avg_con_pt': [mean_scaler.transform([[stats['Mean']]])[0, 0]],
        'cnt_con_pt': [count_scaler.transform([[stats['Count']]])[0, 0]],
        'std_con_pt': [std_scaler.transform([[stats['Std']]])[0, 0]],
        'med_con_pt': [median_scaler.transform([[stats['Median']]])[0, 0]],
        'category_enc': [category_encoder.transform([stats['Category']])[0]],
        'steel_enc': [steel_encoder.transform([steel])[0]],
        'lot_enc': [lot_encoder.transform([lot])[0]]
    })

    # 4. 예측 및 시간 측정
    start_time = time.time()
    prediction = model.predict(input_data)[0]
    elapsed_time = time.time() - start_time

    print("Prediction done!")
    print(f"Prediction time: {elapsed_time:.4f} seconds")  # 예측 시간 출력

    return prediction
