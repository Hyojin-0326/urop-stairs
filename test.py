import numpy as np
import matplotlib.pyplot as plt
import json

# 1️⃣ 파일 경로 설정
meta_file_path = "data/meta.txt"  # 메타데이터 파일 경로
bin_file_path = "data/rgb_data.bin"       # 바이너리 이미지 파일 경로
width, height = 640, 480         # D435i 해상도
frame_count_expected = 27         # 프레임 개수 기대값
dtype = np.uint16                 # Depth라면 uint16, RGB라면 uint8

# 2️⃣ 메타파일 읽어서 프레임 개수 확인
with open(meta_file_path, "r") as f:
    meta_data = json.load(f)

frame_count = meta_data.get("frames", None)

if frame_count is None:
    raise ValueError("메타파일에 'frames' 키가 없습니다.")
if frame_count != frame_count_expected:
    raise ValueError(f"프레임 개수 오류: {frame_count} (기대값: {frame_count_expected})")

print(f"✅ 메타파일 확인 완료: {frame_count} 프레임")

# 3️⃣ 바이너리 데이터 로드
image_data = np.fromfile(bin_file_path, dtype=dtype)

# 4️⃣ 데이터 크기 확인 및 변환 (Depth 또는 RGB 여부 확인)
expected_size_depth = width * height * frame_count
expected_size_rgb = width * height * frame_count * 3  # RGB일 경우

if image_data.size == expected_size_depth:
    image_data = image_data.reshape((frame_count, height, width))
    is_rgb = False
elif image_data.size == expected_size_rgb:
    image_data = image_data.reshape((frame_count, height, width, 3))
    is_rgb = True
else:
    raise ValueError(f"파일 크기 오류: {image_data.size} (예상값: {expected_size_depth} or {expected_size_rgb})")

# 5️⃣ 특정 프레임 시각화
frame_index = 0  # 원하는 프레임 선택
plt.figure(figsize=(8, 5))

if is_rgb:
    plt.imshow(image_data[frame_index])  # RGB 모드
else:
    plt.imshow(image_data[frame_index], cmap="gray")  # Depth 모드

plt.axis("off")
plt.title(f"Frame {frame_index}")
plt.show()
