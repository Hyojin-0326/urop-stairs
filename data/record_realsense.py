#바이너리 저장 코드
################################
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 현재 실행 중인 파일의 디렉토리
script_dir = os.path.dirname(os.path.abspath(__file__))

# 저장할 파일 경로
rgb_save_path = os.path.join(script_dir, 'rgb_data.bin')
depth_save_path = os.path.join(script_dir, 'depth_data.bin')
meta_save_path = os.path.join(script_dir, 'meta.txt')  # 프레임 수 저장

# 파이프라인 생성
pipeline = rs.pipeline()
config = rs.config()

# 해상도 및 프레임 설정
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트리밍 시작
profile = pipeline.start(config)

# 프레임 저장 리스트
rgb_frames = []
depth_frames = []
MAX_FRAMES = 500  # 최대 저장할 프레임 개수

try:
    while True:
        # 프레임 받기
        for _ in range(5):  # 몇 프레임 건너뛰고 저장 (프레임 안정화)
            frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("⚠️ 프레임을 수신하지 못했습니다.")
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # Depth는 float32
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)   # RGB는 uint8
        
        rgb_frames.append(color_image)
        depth_frames.append(depth_image)

        # 화면 출력
        cv2.imshow('RGB Video', color_image)
        cv2.imshow('Depth Video', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        # `q` 키 입력 시 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("🛑 사용자가 'q'를 눌러 녹화를 중단했습니다.")
            break

except KeyboardInterrupt:
    print("🛑 강제 종료 감지됨. 녹화된 데이터를 저장합니다.")

finally:
    # 데이터 저장
    print("💾 데이터 저장 중...")

    # RGB 데이터 저장 (uint8)
    np.array(rgb_frames, dtype=np.uint8).tofile(rgb_save_path)

    # Depth 데이터 저장 (float32)
    np.array(depth_frames, dtype=np.float32).tofile(depth_save_path)

    # 메타데이터 저장 (프레임 개수 기록)
    with open(meta_save_path, 'w') as f:
        f.write(f"{len(rgb_frames)}\n")

    pipeline.stop()
    cv2.destroyAllWindows()
    print("✅ 데이터 저장 완료.")








#피클로 저장하는 코드
###############################################
# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import pickle
# import os


# # 현재 실행 중인 파일의 디렉토리 (스크립트가 있는 폴더)
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # 저장할 경로 설정 (현재 실행 중인 폴더 내부에 저장)
# rgb_save_path = os.path.join(script_dir, 'rgb_data.pkl')
# depth_save_path = os.path.join(script_dir, 'depth_data.pkl')

# # NumPy 버전 문제 해결
# os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# # 파이프라인 생성
# pipeline = rs.pipeline()
# config = rs.config()

# # 해상도 및 프레임 설정
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # 스트리밍 시작
# profile = pipeline.start(config)

# # 프레임 저장 리스트
# rgb_frames = []
# depth_frames = []
# MAX_FRAMES = 500  # 최대 저장할 프레임 개수

# try:
#     while True:
#         # 프레임 받기
#         for _ in range(5):
#             frames = pipeline.wait_for_frames()
        
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
        
#         if not depth_frame or not color_frame:
#             print("⚠️ 프레임을 수신하지 못했습니다.")
#             continue
        
#         depth_image = np.asanyarray(depth_frame.get_data()).copy()
#         color_image = np.asanyarray(color_frame.get_data()).copy()
        
#         # 저장
#         rgb_frames.append(color_image)
#         depth_frames.append(depth_image)

#         # 화면 출력
#         cv2.imshow('RGB Video', color_image)
#         cv2.imshow('Depth Video', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

#         # `q` 키 입력 시 종료
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             print("🛑 사용자가 'q'를 눌러 녹화를 중단했습니다.")
#             break

# except KeyboardInterrupt:
#     print("🛑 강제 종료 감지됨. 녹화된 데이터를 저장합니다.")

# finally:
#     # 데이터 저장
#     print("💾 데이터 저장 중...")
#     with open(rgb_save_path, 'wb') as f:
#         pickle.dump(rgb_frames, f)
#     with open(depth_save_path, 'wb') as f:
#         pickle.dump(depth_frames, f)
    
#     pipeline.stop()
#     cv2.destroyAllWindows()
#     print("✅ 데이터 저장 완료.")
    


