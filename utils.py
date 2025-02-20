import numpy as np
import yolo

def pixel_to_3d(u, v, depth_val, fx, fy, cx, cy):
    X = (u - cx) * depth_val / fx
    Y = (v - cy) * depth_val / fy
    Z = depth_val
    return X, Y, Z
    
def stairs(bbox):
    angle =
    height = 
    distance = 


#-----------
import cv2
import torch
import numpy as np

def segment_stairs_in_roi(color_img, bbox, model, device='cuda'):
    """
    세그멘테이션 모델을 사용하여, bbox 내 계단 영역을 마스킹하는 함수.
    단, bbox를 세로로 3등분한 뒤, 하단 2/3만 세그멘테이션 수행하여 연산량을 줄임.

    Params
    ------
    color_img: np.ndarray
        - 원본 컬러 이미지 (H, W, 3) (BGR 또는 RGB)
    bbox: (x1, y1, x2, y2)
        - YOLO로부터 받은 계단 ROI 좌표
    model: torch.nn.Module (예시)
        - 세그멘테이션 PyTorch 모델(예: U-Net, DeepLab, etc.)
    device: str
        - 'cuda' 또는 'cpu'
    
    Returns
    -------
    mask_full: np.ndarray (H_roi, W_roi)
        - ROI 전체 크기에 대응하는 세그멘테이션 마스크 (0=배경, 1=계단 ...)
        - 상단 1/3 구간은 세그멘테이션을 수행하지 않았으므로 0(배경) 처리됨
    """

    x1, y1, x2, y2 = bbox
    roi_color = color_img[y1:y2, x1:x2]  # ROI 추출

    # --- 1) ROI를 세로로 3등분 ---
    roi_h, roi_w = roi_color.shape[:2]
    one_third = roi_h // 3  # 3등분 높이
    # 하단 2/3 구간: (one_third, roi_h)
    sub_roi_color = roi_color[one_third: , :]

    # --- 2) 모델 입력 전처리(예시) ---
    # 모델에 따라 Resize, Normalize 등이 필요할 수 있음
    # 예: (H, W, 3) BGR -> (1, 3, H, W) RGB 텐서
    sub_roi_rgb = cv2.cvtColor(sub_roi_color, cv2.COLOR_BGR2RGB)
    sub_roi_tensor = torch.from_numpy(sub_roi_rgb).float().permute(2,0,1).unsqueeze(0)  # (1,3,h,w)
    # 간단히 0~1 스케일링
    sub_roi_tensor = sub_roi_tensor / 255.0
    sub_roi_tensor = sub_roi_tensor.to(device)

    # --- 3) 세그멘테이션 수행 ---
    model.eval()
    with torch.no_grad():
        pred = model(sub_roi_tensor)  # (1, num_classes, h, w) 형태 가정
        # 예: 채널 차원에서 argmax -> 클래스 맵
        pred_mask = torch.argmax(pred, dim=1).squeeze(0)  # (h, w)
    
    sub_roi_mask = pred_mask.cpu().numpy().astype(np.uint8)  # 세그멘테이션 결과 (하단 2/3 구간)

    # --- 4) 세그멘테이션 결과를 ROI 전체 크기에 맞춰 합치기 ---
    # 상단 1/3 부분은 세그멘테이션을 수행하지 않았으므로 배경(0)으로 둠
    mask_full = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask_full[one_third: , :] = sub_roi_mask  # 하단 2/3 부분만 예측 결과 반영

    return mask_full


# ----------------------
# 예시: 후속처리 (에지 검출 + 직선 검출) 함수
# ----------------------

def postprocess_stair_mask(mask_full):
    """
    세그멘테이션 마스크에서 계단 윤곽을 좀 더 깔끔하게 얻기 위한 후속 처리 예시
    1) 모폴로지 연산
    2) 에지 검출(Canny)
    3) (선택) HoughLinesP로 계단 엣지 직선 검출
    """

    # 1) 모폴로지 연산으로 잡음 제거/채움
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2) 에지 검출
    # 세그멘테이션 마스크 자체가 0/1(or 0/255)이므로, 
    # 윤곽선을 찾으려면 Canny를 적용하거나, cv2.findContours()도 가능
    edges = cv2.Canny(mask_clean, 50, 150)

    # 3) 직선 검출 (예시)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=10
    )

    # 필요 시 lines_p를 후속 처리(병합/필터링) 후, 각도 계산 등에 활용
    return edges, lines_p


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

    mask_roi = segment_stairs_in_roi(color_img, yolo_bbox, seg_model, device=device)
    
    # 후속처리
    edges, lines_p = postprocess_stair_mask(mask_roi)

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
