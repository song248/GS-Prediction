import cv2
import os
import numpy as np

# 입력 및 출력 폴더 경로
input_folder = 'images'
output_folder = 'pre_img'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 처리 루프
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"이미지를 불러올 수 없습니다: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 전처리 단계
        filtered_image = cv2.bilateralFilter(img_rgb, -1, 10, 10)
        inverted = cv2.bitwise_not(filtered_image)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(inverted, kernel, iterations=1)
        thick_lines = cv2.bitwise_not(dilated)

        # 저장 (RGB를 다시 BGR로 변환 후 저장)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cv2.cvtColor(thick_lines, cv2.COLOR_RGB2BGR))

        print(f"처리 및 저장 완료: {output_path}")
