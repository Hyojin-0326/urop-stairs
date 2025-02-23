import open3d as o3d
import numpy as np
import torch

def depth_to_pointcloud(depth_map):
    """
    Open3D GPU 가속을 활용하여 Depth 이미지를 포인트클라우드로 변환.
    :param depth_map: (H, W) 형태의 NumPy 배열 (Depth 이미지)
    :param intrinsic_matrix: 3x3 형태의 카메라 내적 행렬 (fx, fy, cx, cy 포함)
    :param depth_scale: Depth 값의 스케일링 (RealSense는 1000.0을 사용)
    :return: Open3D PointCloud 객체
    """
    h, w = depth_map.shape

    # ✅ CUDA 연산 없이 바로 Open3D Tensor로 변환 (PyTorch X)
    depth_o3d = o3d.core.Tensor(depth_map.astype(np.float32) / depth_scale, dtype=o3d.core.Dtype.Float32)

    # ✅ Open3D의 Tensor 기반 Intrinsic 설정 (GPU 최적화됨)
    intrinsic_o3d = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # ✅ Open3D GPU 기반 변환 (to_legacy() 사용 안 함)
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic_o3d)

    return pcd  # ✅ Open3D GPU 포인트클라우드 유지
