import cv2
import torch
import numpy as np
from ultralytics import YOLO
import utils
import matplotlib as plt
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import scipy.special  
from scipy.special import expit

#-----------------ultralytics 욜로 객체처럼 변환하는거
class Boxes:
    """ 바운딩 박스 데이터를 YOLO 형식과 유사하게 감싸는 클래스 """
    def __init__(self, bboxes, scores, cls):
        self.xyxy = np.array(bboxes)  # 좌표 정보
        self.scores = np.array(scores)  # 신뢰도 점수
        self.cls = np.array(cls)  # 클래스 ID

class DetectionResult:
    """ YOLO 형식과 호환되도록 boxes 속성을 포함한 결과 객체 """
    def __init__(self, bboxes, scores, cls):
        self.boxes = Boxes(bboxes, scores, cls)  # Boxes 객체로 감싸기




#---------------전역변수모음
class Config:
    current_path = os.path.dirname(os.path.abspath(__file__))
    onnx_path=os.path.join(current_path, "yolo", "yolo_model.onnx")
    model_path = os.path.join(current_path, "yolo", "best.pt")
    trt_path=os.path.join(current_path, "yolo", "yolo_model.trt")



#---------------YOLO----------------
def load_trt_engine(engine_path):
    """TensorRT 엔진 로드"""
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def load_model(model_path=Config.model_path, trt_path=Config.trt_path):
    """YOLO TensorRT 모델 불러오기"""
    trt_path = Config.trt_path
    engine = load_trt_engine(trt_path)
    context = engine.create_execution_context()
    return engine, context


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """ 이미지 비율을 유지하면서 YOLO 입력 크기(640x640)로 패딩 """
    shape = image.shape[:2]  # 현재 (H, W)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 크기 비율 유지
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # 새로운 크기
    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 패딩 추가 (좌우/상하)
    dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
    dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 패딩 적용
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    image_padded = image_padded.astype(np.float32)

     # (H, W, C) → (C, XH, W) 변환 필요할 경우 수행
    if image_padded.shape[-1] == 3:  # 마지막 차원이 3이면 변환 필요
        image_padded = np.transpose(image_padded, (2, 0, 1))  # (H, W, C) → (C, H, W)


    # 배치 차원 추가 (YOLO 모델이 (1, C, H, W) 형식을 기대할 수도 있음)
    image_padded = np.expand_dims(image_padded, axis=0)  # (C, H, W) → (1, C, H, W)

    # 메모리 연속성을 보장
    image_padded = np.ascontiguousarray(image_padded)

    print("C_CONTIGUOUS:", image_padded.flags['C_CONTIGUOUS'])
    print("Shape:", image_padded.shape)  # (1, 3, 640, 640) 형태가 되어야 함
    print("Dtype:", image_padded.dtype)

    return image_padded


def postprocess(output, img_shape, conf_thres=0.2, iou_thres=0.4):
    """ YOLO TensorRT 후처리: 바운딩 박스 & NMS 적용 """
    print("Output shape before processing:", output.shape)
    
    # 1. 배치 차원 제거 (예: (1, 65, 80, 80) -> (65, 80, 80))
    if output.shape[0] == 1:
        output = np.squeeze(output, axis=0)
    print("After squeeze:", output.shape)
    
    # 2. (65, 80, 80) → (80, 80, 65) 변환
    output = output.transpose(1, 2, 0)
    print("After transpose:", output.shape)
    
    # 3. (80, 80, 65) → (6400, 65)로 변환
    output = output.reshape(-1, output.shape[-1])
    print("After reshape:", output.shape)

    num_detections = output.shape[0]  # 총 detection 수
    bboxes = []
    scores = []
    class_ids = []
    
    for i in range(num_detections):
        detection = output[i]

        confidence = float(scipy.special.expit(detection[4]))  # objectness score
        if confidence < conf_thres:
            continue

        class_probs = scipy.special.expit(detection[5:]) 
        class_id = int(np.argmax(class_probs))
        score = float(class_probs[class_id]) * confidence
        if score < conf_thres:
            continue

        x_center, y_center, w, h = detection[0:4] * np.array([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])

        print("Detection values:", detection[0:4])  # 출력 값 확인


        x1 = int(x_center - w / 2)  # x1 = center - width/2
        y1 = int(y_center - h / 2)  # y1 = center - height/2
        x2 = int(x_center + w / 2)  # x2 = center + width/2
        y2 = int(y_center + h / 2)  # y2 = center + height/2
  
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img_shape[1]: x2 = img_shape[1]
        if y2 > img_shape[0]: y2 = img_shape[0]

        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)
    
    # 5. NMS 적용
    indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        final_bboxes = [bboxes[i] for i in indices.flatten()]
        final_scores = [scores[i] for i in indices.flatten()]
        final_class_ids = [class_ids[i] for i in indices.flatten()]
    else:
        final_bboxes, final_scores, final_class_ids = [], [], []

    return final_bboxes, final_scores, final_class_ids


def detect(engine, context, image):
    """ TensorRT 기반 YOLO 추론 (기존 YOLO results 객체처럼 출력) """

    print("DEBUG: detect() 시작")

    if isinstance(image, str):  # 이미지 파일 경로 입력
        image = cv2.imread(image)

    img_shape = image.shape[:2]  # (H, W) 저장
    print("DEBUG: Image shape:", img_shape)

    image_padded = letterbox(image)  # YOLO 입력 크기 맞춤
    image_padded = image_padded.astype(np.float32) / 255.0  # 정규화

    print("Final shape before CUDA:", image_padded.shape)  # ✅ (1, 3, 640, 640) 확인

    # TensorRT 실행
    d_input = cuda.mem_alloc(image_padded.nbytes)
    print("DEBUG: d_input 메모리 할당 완료")

    # ✅ 엔진에서 출력 텐서 크기 가져오기
    output_shape = context.get_binding_shape(1)  # 1번 인덱스가 출력 텐서
    output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize  # 총 바이트 수 계산
    print("DEBUG: Output shape:", output_shape)

    d_output = cuda.mem_alloc(int(output_size))  # ✅ 정확한 크기 설정
    print("DEBUG: d_output 메모리 할당 완료")

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    print("DEBUG: CUDA Stream 생성 완료")

    # 입력 데이터 복사
    cuda.memcpy_htod_async(d_input, image_padded, stream)
    print("DEBUG: 입력 데이터 복사 완료")

    # TensorRT 실행
    try:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        print("DEBUG: TensorRT 실행 완료")
    except Exception as e:
        print("ERROR: TensorRT 실행 중 오류 발생", str(e))
        return []

    # ✅ 출력 데이터를 위한 새로운 배열 생성
    h_output = np.empty(output_shape, dtype=np.float32)  # 모델 출력 크기와 동일한 형태
    print("DEBUG: h_output 배열 생성 완료")

    # ✅ GPU → CPU로 출력 데이터 복사
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    print("DEBUG: GPU → CPU 데이터 복사 완료")

    # ✅ 후처리 함수에서 h_output을 사용하도록 수정
    final_bboxes, final_scores, final_class_ids = postprocess(h_output, img_shape)
    if not final_bboxes:
        print("DEBUG: postprocess has done, No detections found")
        return []

    # 기존 YOLO `results` 객체처럼 변환
    results = [DetectionResult(final_bboxes, final_scores, final_class_ids)]
    
    # ✅ 디버깅 코드 추가
    print("Detections:")
    for i in range(len(results[0].boxes.xyxy)):
        print(f"Box {i}: {results[0].boxes.xyxy[i]}, Score: {results[0].boxes.scores[i]}, Class: {results[0].boxes.cls[i]}")

    print("DEBUG: detect() 완료")
    return results  # 기존 코드와 호환되도록 리스트 형태 반환





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
    save_path = os.path.dirname(os.path.abspath(__file__))
    cv2.imwrite(save_path, img)
    print(f"✅ Bounding box image saved as: {save_path}")

        # 🔥 Matplotlib으로 이미지 표시 (선택)
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("YOLO Detection")
        plt.axis("off")
        plt.show()