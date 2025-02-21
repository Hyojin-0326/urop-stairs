import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import stairs
import os


def pixel_to_3d(u, v, depth_val, fx, fy, cx, cy):
    X = (u - cx) * depth_val / fx
    Y = (v - cy) * depth_val / fy
    Z = depth_val
    return X, Y, Z

#######❗❗❗❗❗ 수정필요, 위아래 픽셀로 높이재는거

def diff_in_height():
    return 0

#----------------- 데이터 로드 함수
def load_rgb_from_bin(bin_path, frame_idx, height=480, width=640):
    data_path = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(data_path, "data", "meta.txt")

    # 🔹 1) meta.txt에서 프레임 개수 읽기
    try:
        with open(meta_path, "r") as f:
            total_frames = int(f.readline().strip())  # 첫 번째 줄에 저장된 프레임 개수 읽기
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 메타데이터 파일 {meta_path}을 찾을 수 없습니다!")
    except ValueError:
        raise ValueError(f"❌ {meta_path}에서 프레임 개수를 읽을 수 없습니다!")

    # 🔹 2) frame_idx가 유효한지 확인
    if frame_idx >= total_frames or frame_idx < 0:
        raise ValueError(f"⚠️ frame_idx {frame_idx}가 유효하지 않습니다! (총 {total_frames}개 프레임)")

    # 🔹 3) .bin 파일에서 RGB 데이터 로드
    try:
        rgb_data = np.fromfile(bin_path, dtype=np.uint8)

        # 전체 데이터가 (total_frames, H, W, 3) 크기인지 확인
        expected_size = total_frames * height * width * 3
        if len(rgb_data) != expected_size:
            raise ValueError(f"❌ RGB 데이터 크기 불일치! 예상 {expected_size}, 실제 {len(rgb_data)}")

        # 🔹 4) (프레임 개수, H, W, 3) 형태로 reshape
        rgb_data = rgb_data.reshape((total_frames, height, width, 3))

        # 🔹 5) frame_idx에 해당하는 프레임 반환
        rgb_image = rgb_data[frame_idx]

    except Exception as e:
        raise RuntimeError(f"❌ RGB .bin 파일을 로드하는 중 오류 발생: {e}")

    return rgb_image



def align_depth_to_rgb(depth_bin_path, rgb_bin_path, frame_idx, height=480, width=640):
    context = rs.context()
    devices = context.query_devices()
    data_path = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(data_path, "data", "meta.txt")
    try:
        with open(meta_path, "r") as f:
            total_frames = int(f.readline().strip())  # 첫 줄에서 프레임 개수 읽기
        print(f"🔹 meta.txt에서 읽은 프레임 개수: {total_frames}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ 메타데이터 파일 {meta_path}을 찾을 수 없습니다!")
    except ValueError:
        raise ValueError(f"❌ {meta_path}에서 프레임 개수를 읽을 수 없습니다!")

    if len(devices) == 0:
        print("🔹 No device connected, using default intrinsics & loading from .bin files")
        intrinsics = rs.intrinsics()
        intrinsics.width = width
        intrinsics.height = height
        intrinsics.ppx = 308.5001  # 기본 광학 중심 X (cx)
        intrinsics.ppy = 246.4238  # 기본 광학 중심 Y (cy)
        intrinsics.fx = 605.9815  # 기본 초점 거리 X (fx)
        intrinsics.fy = 606.1337  # 기본 초점 거리 Y (fy)
        intrinsics.model = rs.distortion.none  # 왜곡 없음
        intrinsics.coeffs = [0, 0, 0, 0, 0]  
        
        # --- .bin 파일에서 RGB & Depth 불러오기 ---
        depth_map = np.fromfile(depth_bin_path, dtype=np.float32)
        depth_map = depth_map.reshape((total_frames, height, width))
        depth_map = depth_map[frame_idx]

        rgb_image = load_rgb_from_bin(rgb_bin_path, frame_idx)

        if frame_idx >= total_frames:
            raise ValueError(f"⚠️ frame_idx {frame_idx}가 저장된 프레임 개수 {total_frames}보다 큼")

    else:
        try:
            print("✅ Realsense device detected, capturing frames...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)

            # 카메라 Intrinsics 가져오기
            color_profile = profile.get_stream(rs.stream.color)
            intr = color_profile.as_video_stream_profile().get_intrinsics()
            fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

            # Depth → RGB 정렬 수행
            align_to = rs.stream.color
            align = rs.align(align_to)

            # 프레임 수집 및 정렬
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("⚠️ Failed to capture frames from Realsense.")

            # numpy 배열 변환
            depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            rgb_image = np.asanyarray(color_frame.get_data())

            pipeline.stop()

        except RuntimeError:
            print("No device connected (error during capture), using default intrinsics")
            profile = None
            depth_map = np.zeros((height, width), dtype=np.float32)  # 빈 Depth 맵 생성
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)  # 빈 RGB 이미지 생성

    return depth_map, rgb_image

#--------중복박스 있는지 체크 
def check_duplicate(results):
    seen = set()
    duplicates = False
    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in seen:
                    duplicates = True
                else:
                    seen.add(cls_id)
    return duplicates # 불리안 아웃풋임

#------- 중복박스 없애고 가까운것만 반환
def remove_extra_box(results, depth_map):
    all_boxes = []  # 모든 바운딩 박스를 저장할 리스트

    # YOLO 탐지 결과에서 모든 바운딩 박스 수집 (클래스 ID 포함)
    for result in results:
        for box in result.boxes:
            bbox = tuple(map(int, box.xyxy[0]))  # 바운딩 박스 좌표 (x1, y1, x2, y2)
            cls_id = int(box.cls[0])  # 클래스 ID 가져오기
            all_boxes.append((cls_id, bbox))  # (클래스 ID, 바운딩 박스) 저장

    # 탐지된 객체가 없으면 (None, None) 반환
    if not all_boxes:
        return (None, None)

    # 가장 가까운 객체 찾기
    return get_closest_box_with_depth(all_boxes, depth_map)


def get_closest_box_with_depth(boxes, depth_map):
    """ 가장 가까운 바운딩 박스를 선택 (최소 Depth 값 기준) """
    min_depth = float("inf")
    closest_box = None
    closest_cls_id = None

    for cls_id, bbox in boxes:
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]  

        # 0이 아닌 Depth 값이 있으면 최솟값 계산 
        valid_depths = roi_depth[roi_depth > 0]
        if len(valid_depths) > 0:
            min_roi_depth = np.min(valid_depths)  
            if min_roi_depth < min_depth:
                min_depth = min_roi_depth
                closest_box = bbox
                closest_cls_id = cls_id  # 가장 가까운 박스의 클래스 ID 저장

    return (closest_cls_id, closest_box) 




#-----------바운딩박스의 ROI 크롭하기
def crop_roi(bbox, rgb_image, depth_map):
    x1,y1,x2,y2 = bbox
    rgb_roi = rgb_image[y1:y2, x1:x2, :]
    depth_roi = depth_map[y1:y2, x1:x2]
    return rgb_roi, depth_roi


#--------- 클래스 분류해서 함수 실행 ❗❗❗❗❗❗❗❗❗❗수정필요
def measure_height(cls_id,rgb_roi, depth_roi, bbox,model):
    if cls_id ==0:
        stairs.measure_height(rgb_roi, depth_roi, bbox, model)
    #❗디버깅용
    print("높이 측정중")









# ----------------------
# 실제 사용 예시
# ----------------------
if __name__ == "__main__":
    # 가정: color_img (BGR)와 YOLO로부터 얻은 bbox, 세그멘테이션 모델 준비
    color_img = cv2.imread("test.jpg")
    yolo_bbox = (100, 200, 400, 500)  # 예시 (x1, y1, x2, y2)

    # 예: 임의의 세그멘테이션 모델 (PyTorch)
    # model = MySegModel(...)
    # model.load_state_dict(torch.load("model.pth"))
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)

    # 여기서는 데모라서 model 대신 None 처리
    model = None
    device = 'cpu'

    # 세그멘테이션 함수 시연 (실제로는 model이 필요)
    # segment_stairs_in_roi 함수 내 model 부분을 직접 수정해서 사용 가능
    # 혹은 아래처럼 "더미"로 예시를 만들 수도 있음
    def dummy_model(x):
        # 입력 x: (1,3,h,w)
        # 가짜로 전부 '1' 클래스라고 치자 (전부 계단)
        return torch.zeros((1, 2, x.shape[2], x.shape[3]), device=x.device) + 0.5

    seg_model = dummy_model

    mask_roi = stairs.segment_stairs_in_roi(color_img, yolo_bbox, seg_model, device=device)
    
    # 후속처리
    edges, lines_p = stairs.postprocess_stair_mask(mask_roi)

    # 시각화 예시
    # ROI 영역만 시각화
    x1, y1, x2, y2 = yolo_bbox
    roi_vis = color_img[y1:y2, x1:x2].copy()

    # 에지 그리기
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[edges != 0] = (0, 0, 255)  # 빨간색

    # lines_p가 있으면 직선 시각화
    if lines_p is not None:
        for line in lines_p:
            x_start, y_start, x_end, y_end = line[0]
            cv2.line(roi_vis, (x_start, y_start), (x_end, y_end), (0,255,0), 2)

    # 결과 보기
    cv2.imshow("ROI mask", mask_roi*255)  # 0 or 1 => 시각화를 위해 255 곱
    cv2.imshow("ROI edges", edges_bgr)
    cv2.imshow("ROI lines", roi_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
