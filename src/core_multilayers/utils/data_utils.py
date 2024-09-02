""" Tools for data processing.
    Author: chenxi-wang
"""

import numpy as np
import cv2

class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)

    # 代表图像中每个像素的横向和纵向坐标
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)

    # np.meshgrid 函数用于生成网格点坐标矩阵。
    # xmap 和 ymap 被转换成两个二维数组，其中每个元素分别代表对应像素位置的 x 和 y 坐标
    xmap, ymap = np.meshgrid(xmap, ymap)

    # 计算每个像素点的实际深度值。
    # 这里假设 depth 数组中的值是通过某种比例因子 camera.scale 缩放的，这行代码将其转换回真实的深度值
    points_z = depth / camera.scale
    
    # 通过已有的像素坐标(xmap, ymap)，计算每个点的 x 和 y 世界坐标
    # 像素坐标减去光心点，乘以深度值 并除以焦距
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def create_point_cloud_from_depth_image_dict(depth, camera, organized=True):

    assert (depth.shape[0] == camera['height'] and depth.shape[1] == camera['width'])
    xmap = np.arange(camera['width'])
    ymap = np.arange(camera['height'])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth
    points_x = (xmap - camera['cx']) * points_z / camera['fx']
    points_y = (ymap - camera['cy']) * points_z / camera['fy']
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    
    # 从点云中选取那些分割标签大于 0 的点 (默认是全选)
    foreground = cloud[seg > 0]

    # 计算前景点云在 x、y、z 方向上的最小和最大值，这些值定义了一个轴对齐的包围盒
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)

    # 分别为 x、y、z 坐标创建掩码，检查点云每个点的坐标是否在前景点云构造的包围盒扩展了 outlier 值的范围内
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    # 将三个方向上的掩码进行逻辑与操作，得到一个总的掩码，表示每个点是否在给定的mask扩展的工作区域内。
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

def interp(depth, x, y):
    cx = np.floor(x).astype(int)

    cy = np.floor(y).astype(int)


    a = cx + 1 - x
    b = x - cx
    c = cy + 1 - y
    d = y - cy

    f1 = depth[cy, cx]
    f2 = depth[cy, cx + 1]
    f3 = depth[cy + 1, cx]
    f4 = depth[cy + 1, cx + 1]


    fx1 = a * f1 + b * f2
    fx2 = a * f3 + b * f4
    fy = c * fx1 + d * fx2

    return fy

def check_grasp_object_material(grasp_point, sim):
    material_mask_path =sim.render_root + f"/{str(sim.ann_id).zfill(4)}/mask/{str(sim.ann_id).zfill(4)}.png"
    material_mask = cv2.imread(material_mask_path)
    K = sim.intrinsic.sim_d415_rgb_old()
    grasp_point_proj = np.dot(K, grasp_point.T * 1000)
    grasp_point_proj = grasp_point_proj / grasp_point_proj[2]
    grasp_point_proj = grasp_point_proj[0:2].astype(int)
    return material_mask[grasp_point_proj[1]][grasp_point_proj[0]][0]

def check_grasp_object_id(grasp_point, sim):
    K = sim.intrinsic.sim_d415_rgb_old()
    grasp_point_proj = np.dot(K, grasp_point.T * 1000)
    grasp_point_proj = grasp_point_proj / grasp_point_proj[2]
    grasp_point_proj = grasp_point_proj[0:2].astype(int)
    _,_,seg = sim.get_camera_image()
    if seg[grasp_point_proj[1]][grasp_point_proj[0]] != 0:
        return sim.object_list[seg[grasp_point_proj[1]][grasp_point_proj[0]]-1]

    return -1
