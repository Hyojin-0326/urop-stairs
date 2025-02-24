import torch
import cv2
import numpy as np
import utils
import open3d as o3d



#----------- 파이프라인
def measure_height(depth_roi):
    pcld_np = np.asarray(pcld.points)
    pcld = utils.extract_plane_ransac(depth_roi)
    if not pcld.has_normals():
        pcld.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    plane_normals = np.asarray(pcld.normals)
    avg_normal = np.mean(plane_normals, axis=0)

    angle = calculate_angle(avg_normal)
    height=calculate_height(pcld_np)



    # 첫 번째 점의 좌표 출력
    print(f"첫 번째 3D 좌표: {pcld_np[0]}")
    print(f"포인트 클라우드의 총 점 개수: {pcld_np.shape[0]}")
    print(f'{angle} 라디안, {height} 미터')
    return angle, height


def calculate_angle(normal_vector):
    # 수평면의 법선 벡터 (z 축 방향)
    horizontal_normal = np.array([0, 0, 1])
    
    # 법선 벡터와 수평면 벡터 사이의 내적 계산
    dot_product = np.dot(normal_vector, horizontal_normal)
    
    # 각도 계산 (radian 단위)
    angle_rad = np.arccos(dot_product / (np.linalg.norm(normal_vector) * np.linalg.norm(horizontal_normal)))
    
    return angle_rad

def calculate_height(pcld_np):
    
    pcld_sorted = pcld_np[np.argsort(pcld_np[:, 2])]  # z 값 기준으로 오름차순 정렬
    
    # 윗쪽 엣지 (상위 10%의 포인트들)
    upper_edge = pcld_sorted[int(0.9 * len(pcld_sorted)):]  # 90% 이상은 윗쪽 엣지
    
    # 아랫쪽 엣지 (하위 10%의 포인트들)
    lower_edge = pcld_sorted[:int(0.1 * len(pcld_sorted))]  # 10% 이하를 아랫쪽 엣지
    
    # 윗쪽 엣지와 아랫쪽 엣지의 평균 z 값을 계산
    upper_edge_mean = np.mean(upper_edge[:, 2])
    lower_edge_mean = np.mean(lower_edge[:, 2])
    
    # 높이는 윗쪽 엣지의 평균 높이에서 아랫쪽 엣지의 평균 높이를 빼서 계산
    height = upper_edge_mean - lower_edge_mean
    
    return height