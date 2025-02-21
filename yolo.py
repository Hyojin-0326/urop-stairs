import cv2
import torch
import numpy as np
from ultralytics import YOLO
import utils
import matplotlib as plt

#---------------YOLO----------------
def load_model(model_path):
    """ 학습된 YOLO 모델을 불러옴 """
    model = YOLO(model_path)  # YOLO 모델 불러오기
    print(model.names)  # 클래스 목록 출력
    return model

def detect(model, image):
    if isinstance(image,str):
        #results = model.predict(image, device = "cuda")
        results = model.predict(image)
    elif isinstance(image, np.ndarray):
        results = model.predict(image)
        #results = model.predict(image, device = 'cuda')
        
    else:
        raise ValueError("image는 파일 경로나 np array여야 함.")
    for result in results:
        print(result)
    return results






#--------걍 테스트용-----------



def main():
    model_path = "best.pt"
    image_path = "test.jpg"

    model = YOLO(model_path)  # YOLO 모델 로드
    results = model(image_path)  # 객체 탐지 실행

    print(results)  # 결과 출력 (디버깅용)

if __name__ == "__main__":
    main()

# -----창고---------

def draw_bbox(model, image, show = False):
    """
    YOLO 모델을 사용하여 바운딩 박스를 그리고 결과를 출력.
    
    Parameters
    ----------
    model : YOLO 객체
    image : np.ndarray 또는 str
        - np.ndarray: OpenCV 이미지 (BGR)
        - str: 이미지 파일 경로
    """
    if isinstance(image, str):
        img = cv2.imread(image)  # 파일 경로일 경우 읽기
    elif isinstance(image, np.ndarray):
        img = image.copy()  # numpy 배열일 경우 그대로 사용
    else:
        raise ValueError("image는 파일 경로나 numpy 배열이어야 합니다.")

    results = model.predict(img)  # YOLO 실행
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    ##save path
    save_path = "/home/hjkwon/urop-stairs/data/detected.jpg"
    cv2.imwrite(save_path, img)
    print(f"✅ Bounding box image saved as: {save_path}")

        # 🔥 Matplotlib으로 이미지 표시 (선택)
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("YOLO Detection")
        plt.axis("off")
        plt.show()