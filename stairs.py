import torch
import cv2
import numpy as np
import utils

#----------- 파이프라인
def measure_height(rgb_roi, depth_roi, bbox,model):
    
    segmented_img = segment_stairs_in_roi(rgb_roi,bbox, model)
    detect_structure = postprocess_stair_mask(segmented_img)


    ## 구조 보고 엣지부분만 3차원 좌표로 변환
    pointcloud = utils.pixel_to_3d()
    
    ##픽셀 높이차이 재기
    height = utils.diff_in_height()
    



























#-----------
def segment_stairs_in_roi(color_img, bbox, model, device):
    x1, y1, x2, y2 = bbox
    roi_color = color_img[y1:y2, x1:x2]  # ROI 추출

    # --- 1) ROI를 세로로 3등분 ---
    roi_h, roi_w = roi_color.shape[:2]
    one_third = roi_h // 3  # 3등분 높이
    sub_roi_color = roi_color[one_third: , :]

    # --- 2) 모델 입력 전처리 ---
    sub_roi_rgb = cv2.cvtColor(sub_roi_color, cv2.COLOR_BGR2RGB)
    sub_roi_tensor = torch.from_numpy(sub_roi_rgb).float().permute(2,0,1).unsqueeze(0).to(device)  # (1,3,h,w)
    sub_roi_tensor /= 255.0

    # --- 3) TensorRT 엔진을 통한 세그멘테이션 추론 ---
    with torch.no_grad():
        pred = model(sub_roi_tensor)  # (1, num_classes, h, w) 가정
        pred_mask = torch.argmax(pred, dim=1).squeeze(0)  # (h, w)
    
    sub_roi_mask = pred_mask.cpu().numpy().astype(np.uint8)

    # --- 4) 결과를 원본 ROI 크기로 복원 ---
    mask_full = np.zeros((roi_h, roi_w), dtype=np.uint8)
    mask_full[one_third: , :] = sub_roi_mask

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
