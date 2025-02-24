import os
import torch
from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(current_path, "yolo_model.onnx")
trt_path = os.path.join(current_path, "yolo_model.trt")
model_path = os.path.join(current_path, "best.pt")

# ONNX 변환 함수
def convert_yolo_to_onnx():
    print(f"🔹 [ONNX 변환 시작] 모델 로드 중: {model_path}")
    
    model = YOLO(model_path)
    model.model.to("cuda")
    model.eval()

    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    
    print("🔹 [ONNX 변환 진행 중] torch.onnx.export 실행")
    
    torch.onnx.export(
        model.model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["images"],
        output_names=["output"]
    )
    
    print(f"✅ ONNX 변환 완료: {onnx_path}")
    return onnx_path

# TensorRT 변환 함수
def convert_onnx_to_trt():
    print(f"🔹 [TensorRT 변환 시작] ONNX 모델: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print("❌ ONNX 모델이 존재하지 않음! 먼저 ONNX 변환 실행 필요")
        return

    # TensorRT 변환 실행
    os.system(f"trtexec --onnx={onnx_path} --saveEngine={trt_path} --fp16")
    
    if os.path.exists(trt_path):
        print(f"✅ TensorRT 변환 완료: {trt_path}")
    else:
        print("❌ TensorRT 변환 실패!")

# 변환 실행
if __name__ == "__main__":
    convert_yolo_to_onnx()
    convert_onnx_to_trt()
