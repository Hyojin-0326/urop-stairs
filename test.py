import onnx

onnx_model_path = "yolo/yolo_model.onnx"

# ONNX 모델 로드
model = onnx.load(onnx_model_path)

# 모델 검증
onnx.checker.check_model(model)
print("✅ ONNX 모델이 정상적으로 변환되었습니다!")