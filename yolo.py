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

# def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
#     """ 이미지 비율을 유지하면서 YOLO 입력 크기(640x640)로 패딩 """
#     shape = image.shape[:2]  # 현재 (H, W)
#     ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 크기 비율 유지
#     new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # 새로운 크기
#     image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

#     # 패딩 추가 (좌우/상하)
#     dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
#     dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

#     # 패딩 적용
#     image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#     image_padded = image_padded.astype(np.float32)
#     image_padded = np.ascontiguousarray(image_padded)


#     print("C_CONTIGUOUS:", image_padded.flags['C_CONTIGUOUS'])
#     print("Shape:", image_padded.shape)
#     print("Dtype:", image_padded.dtype)


#     return image_padded


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


def postprocess(output, img_shape, conf_thres=0.5, iou_thres=0.4):
    """ YOLO TensorRT 후처리: 바운딩 박스 & NMS 적용 """
    print("Output shape before processing:", output.shape)

    #output에서 첫 번째 차원을 제거
    #두 번째 차원(60)의 클래스 확률에 접근(어차피 binary 탐색이라 클래스는 1개밖에 없음)

    num_detections = output.shape[0]  # 감지된 객체 개수
    bboxes = []
    scores = []
    class_ids = []

    for i in range(num_detections):
        confidence = float(output[i, 4])  # 객체 신뢰도
        if confidence < conf_thres:
            continue  # 신뢰도 낮으면 무시

        # 클래스 확률 중 가장 높은 것 찾기
        class_probs = output[i, 5:]  # 클래스 확률 (80개)
        class_id = np.argmax(class_probs)
        score = class_probs[class_id] * confidence  # 최종 신뢰도

        if score < conf_thres:
            continue

        # YOLO는 정규화된 좌표를 반환하므로 원본 이미지 크기로 변환
        x_center, y_center, w, h = output[i, :4] * np.array([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        bboxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(class_id)

    # NMS 적용
    indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_thres, iou_thres)
    final_bboxes = [bboxes[i] for i in indices.flatten()]
    final_scores = [scores[i] for i in indices.flatten()]
    final_class_ids = [class_ids[i] for i in indices.flatten()]

    return final_bboxes, final_scores, final_class_ids


class DetectionResult:
    """YOLO results 객체처럼 동작하는 클래스"""
    def __init__(self, bboxes, scores, class_ids):
        self.boxes = np.array(bboxes, dtype=np.float32)  # 바운딩 박스 (xyxy)
        self.scores = np.array(scores, dtype=np.float32)  # 신뢰도 점수
        self.class_ids = np.array(class_ids, dtype=np.int32)  # 클래스 ID

    def __getitem__(self, idx):
        """ 리스트처럼 인덱싱 가능하도록 설정 """
        return (self.boxes[idx], self.scores[idx], self.class_ids[idx])

    def __len__(self):
        return len(self.boxes)

def detect(engine, context, image):
    """ TensorRT 기반 YOLO 추론 (기존 YOLO results 객체처럼 출력) """

    if isinstance(image, str):  # 이미지 파일 경로 입력
        image = cv2.imread(image)

    img_shape = image.shape[:2]  # (H, W) 저장
    image_padded = letterbox(image)  # YOLO 입력 크기 맞춤
    image_padded = image_padded.astype(np.float32) / 255.0  # 정규화

    print("Final shape before CUDA:", image_padded.shape)  # ✅ (1, 3, 640, 640) 확인

    # TensorRT 실행
    d_input = cuda.mem_alloc(image_padded.nbytes)

    # ✅ 엔진에서 출력 텐서 크기 가져오기
    output_shape = context.get_binding_shape(1)  # 1번 인덱스가 출력 텐서
    output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize  # 총 바이트 수 계산
    d_output = cuda.mem_alloc(int(output_size))  # ✅ 정확한 크기 설정

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    # 입력 데이터 복사
    cuda.memcpy_htod_async(d_input, image_padded, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # ✅ 출력 데이터를 위한 새로운 배열 생성
    h_output = np.empty(output_shape, dtype=np.float32)  # 모델 출력 크기와 동일한 형태

    # ✅ GPU → CPU로 출력 데이터 복사
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # ✅ 후처리 함수에서 h_output을 사용하도록 수정
    final_bboxes, final_scores, final_class_ids = postprocess(h_output, img_shape)

    # 기존 YOLO `results` 객체처럼 변환
    results = [DetectionResult(final_bboxes, final_scores, final_class_ids)]

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