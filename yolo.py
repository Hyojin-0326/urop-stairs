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
        results = model(image)
    elif isinstance(image, np.ndarray):
        results = model.predict(image)
    else:
        raise ValueError("image는 파일 경로나 np array여야 함.")
    for result in results:
        print(result)
    return results




def remove_redundant_boxes(results, depth_map):
    """
    YOLO 탐지 결과에서 각 클래스별로 가장 가까운 바운딩 박스 하나만 남김.
    - results: YOLO 탐지 결과 (ultralytics YOLO output)
    - depth_map: Depth 정보가 포함된 numpy 배열
    반환값:
    - {클래스 ID: 가장 가까운 바운딩 박스} 형태의 딕셔너리
    """
    class_boxes = {}  # {클래스 ID: 바운딩 박스 리스트}

    # YOLO 탐지 결과를 클래스별로 정리
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # 클래스 ID 가져오기
            bbox = tuple(map(int, box.xyxy[0]))  # 바운딩 박스 좌표 (x1, y1, x2, y2)

            if cls not in class_boxes:
                class_boxes[cls] = []  # 클래스 ID가 없으면 리스트 생성
            
            class_boxes[cls].append(bbox)  # 해당 클래스의 바운딩 박스 추가

    # 클래스별로 가장 가까운 바운딩 박스만 남기기
    filtered_roi = {}

    for cls, boxes in class_boxes.items():
        closest_box = get_closest_box_with_depth(boxes, depth_map)  # 가장 가까운 객체 선택
        if closest_box:
            filtered_roi[cls] = closest_box  # 필터링된 결과 저장

    return filtered_roi # {클래스 ID: 가장 가까운 바운딩 박스}


def get_closest_box_with_depth(boxes, depth_map):
    """ Depth Map을 이용해 가장 가까운 바운딩 박스 선택 """
    min_depth = float("inf")
    closest_box = None

    for bbox in boxes:
        x1, y1, x2, y2 = bbox

        # ROI(Region of Interest) 설정
        roi_depth = depth_map[y1:y2, x1:x2]

        # Depth 값이 0이 아닌 것들만 평균 계산 (일부 센서는 0이 없는 데이터)
        valid_depths = roi_depth[roi_depth > 0]

        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
            if avg_depth < min_depth:  # 더 가까운 객체라면 갱신
                min_depth = avg_depth
                closest_box = bbox

    return closest_box


def measure_height(filtered_roi):
    height = {}

    for cls_id in filtered_roi.keys():
        if cls_id == 0:  # 계단 높이 측정
            bbox = filtered_roi.get(cls_id)
            if bbox:
                height[cls_id] = utils.stairs(bbox)

        elif cls_id == 1:  
            bbox = filtered_roi.get(cls_id)
            if bbox:
                height[cls_id] = utils.some_other_height_function(bbox)

    return height


















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