import os
import sys
import numpy as np
import torch
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from graspnetAPI.graspnet_eval import GraspGroup
sys.path.append("./gsnet/pointnet2")
sys.path.append("./gsnet/utils")
sys.path.append("./src/core_multilayers")
from gsnet.models.graspnet import GraspNet, pred_decode
from raft_mvs_multilayers import RAFTMVS_2Layer
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from D415_camera import CameraMgr

SHOW_3D = False
SHOW_2D = True

def load_image(imfile, ratio=0.5):
    img = cv2.imread(imfile, 1)     # (1,3,360,640)
    img = img[:, :, ::-1].copy()    # reverse bgr to rgb
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

def load_projmatrix_render_d415():
    cameraMgr = CameraMgr()
    calib_rgb = cameraMgr.sim_d415_rgb()
    calib_ir = cameraMgr.sim_d415_ir()
    pose_rgb = cameraMgr.sim_d415_rgb_pose()
    pose_ir1 = cameraMgr.sim_d415_ir1_pose()
    pose_ir2 = cameraMgr.sim_d415_ir2_pose()

    depth_min = torch.tensor(0.2)
    depth_max = torch.tensor(1.50)

    poses = [pose_rgb, pose_ir1, pose_ir2] # (3,4,4)
    intrinsics = [calib_rgb, calib_ir, calib_ir] # (3,3,3)

    poses = np.stack(poses, 0).astype(np.float32)
    intrinsics = np.stack(intrinsics, 0).astype(np.float32)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    proj = poses.clone()
    proj[:, :3, :4] = torch.matmul(intrinsics, poses[:, :3, :4])

           # (1,3,4,4)
    return proj[None].cuda(), depth_min[None].cuda(), depth_max[None].cuda()


class MVSGSNetEval():
    def __init__(self, cfgs):
        self.args = cfgs
        # 创建 GSNet
        net = torch.nn.DataParallel(GraspNet(seed_feat_dim=cfgs.seed_feat_dim, graspness_threshold=cfgs.graspness_threshold, is_training=False))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.module
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

        # Init the mvs model
        mvs_net = torch.nn.DataParallel(RAFTMVS_2Layer(cfgs)).cuda()
        mvs_net.load_state_dict(torch.load(cfgs.restore_ckpt))
        mvs_net = mvs_net.module

        batch_interval = 100
        mvs_net.eval()
        net.eval()

        # 获取 D415 相机参数
        intrinsic = CameraMgr().sim_d415_rgb()
        # Camera intrisics for point cloud creation.
        self.camera = CameraInfo(640.0, 360.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], 1.0)

        self.mvs_net = mvs_net
        self.gsnet = net

        self.proj_matrices, self.dmin, self.dmax = load_projmatrix_render_d415()

    def project_transform(self, vertices):
        vertices_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1))]) # (N, 4)
        vertices_2d_hom = np.dot(self.proj_matrices.squeeze()[0, :3, :4].cpu().numpy(), vertices_hom.T) # (3, N)
        vertices_2d = vertices_2d_hom[:2, :] / vertices_2d_hom[2, :]   # (2, N)
        return vertices_2d.T.astype(int)   # (N, 2)

    def infer(self, rgb_path, ir1_path, ir2_path, mask_path=None, zrot=None):
        with torch.no_grad():
            color = load_image(rgb_path)
            ir1 = load_image(ir1_path)
            ir2 = load_image(ir2_path)

            # 获得估计的深度 depth_up: (1,2,360,640)
            depth_up = self.mvs_net(color, ir1, ir2, self.proj_matrices.clone(), self.dmin, self.dmax, iters=self.args.valid_iters, test_mode=True)
            depth_up = (depth_up).squeeze() # (2,360,640)
            depth_2layer = depth_up.detach().cpu().numpy().squeeze() # (2,360,640)

            depth = depth_2layer[0] # (360,640)
            cloud = create_point_cloud_from_depth_image(depth, self.camera, organized=True) # (360,640,3)
            depth_mask = (depth>0.25) & (depth<1.0)

            if mask_path is None:
                seg = np.ones(depth.shape)
            else:
                seg = cv2.imread(mask_path, -1)

            workspace_mask = get_workspace_mask(cloud, seg, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)

            cloud_masked = cloud[mask] # (360,640,3)
            color = (color).squeeze() # (3,360,640)
            color = color.detach().cpu().numpy().squeeze()  # (3,360,640)
            color = np.transpose(color, [1,2,0]) # (360,640,3)
            color_masked = color[mask]
            idxs = np.random.choice(len(cloud_masked), self.args.num_point, replace=False)

            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]

            #add second layer
            depth1 = depth_2layer[1]    # (360, 640)
            cloud1 = create_point_cloud_from_depth_image(depth1, self.camera, organized=True)   # (360, 640, 3)
            depth_mask1 = (depth1 > 0.25) & (depth1 < 1.0) & (depth1-depth>0.01)
            mask1 = (depth_mask1 & workspace_mask)
            cloud_masked1 = cloud1[mask1]
            comp_num_pt = 10000
            if (len(cloud_masked1) > 0):
                print('Layer 2 completed_point_cloud : ', len(cloud_masked1))
                if len(cloud_masked1) >= (comp_num_pt):
                    idxs = np.random.choice(len(cloud_masked1), comp_num_pt, replace=False)
                else:
                    idxs1 = np.arange(len(cloud_masked1))
                    idxs2 = np.random.choice(len(cloud_masked1), comp_num_pt - len(cloud_masked1),
                                             replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                completed_sampled = cloud_masked1[idxs]

                # 两层点云和起来 (都经过sample的， self.args.num_point + 10000)
                all_cloud_sampled = np.concatenate([cloud_sampled, completed_sampled], axis=0)
                all_color_sampled = np.concatenate([color_sampled, np.ones((comp_num_pt, 3), dtype=int) * 255], axis=0)
                # completion points colud not be used as graspness point , default setting
                # 第二层点云是无法用于抓取的
                objectness_label = np.concatenate([np.ones([self.args.num_point, ]), (-1)*np.ones([comp_num_pt, ])], axis=0)
                
                # 这个 mask 的点云并没有 sample
                cloud_masked = np.concatenate([cloud_masked, cloud_masked1], axis=0)
            else:
                objectness_label = np.ones([self.args.num_point, ])

            ret_dict = {'point_clouds': all_cloud_sampled.astype(np.float32),
                        'coors': all_cloud_sampled.astype(np.float32) / 0.005,
                        'feats': np.ones_like(all_cloud_sampled).astype(np.float32),
                        'full_point_clouds': cloud_masked.astype(np.float32),
                        'objectness_label': objectness_label.astype(np.int32),
                        }

            batch_data = minkowski_collate_fn([ret_dict])
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()
            # Forward pass
            with torch.no_grad():
                try:
                    end_points = self.gsnet(batch_data)
                except:
                    return None
                grasp_preds = pred_decode(end_points)   # grasp_preds

            torch.cuda.empty_cache()
            preds = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # 防碰撞
            if self.args.collision_thresh > 0:
                cloud = ret_dict['full_point_clouds']
                cloud = cloud_masked.astype(np.float32)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.args.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
                gg = gg[~collision_mask]

            # 减少重叠或接近重叠的抓取姿态
            # 两个抓取之间的平移距离是否足够小和判断两个抓取之间的旋转差异是否足够小，以认为它们是重叠的
            gg = gg.nms()
            gg = gg.sort_by_score()
            count = gg.__len__()
            if count < 1:
                return None
            if count > 5:
                count = 5
            # gg = gg[-1:]
            gg = gg[:count]

            if SHOW_2D:
                # image = cv2.imread(rgb_path)

                # grippers = gg.to_open3d_geometry_list()
                
                # for gripper_mesh in grippers:
                #     gripper_points = np.asarray(gripper_mesh.vertices)
                #     triangles = np.asarray(gripper_mesh.triangles)
                #     gripper_points_2d = self.project_transform(gripper_points)

                #     # 生成随机颜色
                #     # color = tuple(np.random.randint(0, 255, size=3).tolist())
                #     color = (219,112,147)
                #     # import ipdb; ipdb.set_trace()
                #     for tri in triangles:
                #         pts = np.array([
                #             gripper_points_2d[tri[0]],
                #             gripper_points_2d[tri[1]],
                #             gripper_points_2d[tri[2]]
                #         ], dtype=np.int32)

                #         cv2.fillPoly(image, [pts], color=color)

                #         boundary_color = (int(color[0]*0.8), int(color[1]*0.8), int(color[2]*0.8))
                #         cv2.polylines(image, [pts], isClosed=True, color=boundary_color, thickness=1)

                #     point = gripper_points_2d[-1]
                #     # 确保point是整数类型的元组，例如 (x, y)
                #     center = (int(point[0]), int(point[1]))
                #     color = (255, 0, 255)  # 紫色
                #     radius = 3  # 点的半径
                #     thickness = -1  # 填充点
                #     # 在图像上绘制点
                #     cv2.circle(image, center, radius, color, thickness)

                image = cv2.imread(rgb_path)
                grippers = gg.to_open3d_geometry_list()
                gripper_index = 0  # 用于递增数字标注

                for gripper_mesh in grippers:
                    gripper_points = np.asarray(gripper_mesh.vertices)
                    triangles = np.asarray(gripper_mesh.triangles)
                    gripper_points_2d = self.project_transform(gripper_points)

                    color = (0,255,127)
                    for tri in triangles:
                        pts = np.array([
                            gripper_points_2d[tri[0]],
                            gripper_points_2d[tri[1]],
                            gripper_points_2d[tri[2]]
                        ], dtype=np.int32)

                        cv2.fillPoly(image, [pts], color=color)
                        boundary_color = (int(color[0]*0.8), int(color[1]*0.8), int(color[2]*0.8))
                        cv2.polylines(image, [pts], isClosed=True, color=boundary_color, thickness=1)

                    point = gripper_points_2d[-1]
                    center = (int(point[0]), int(point[1]))
                    color = (255, 0, 255)
                    radius = 3
                    cv2.circle(image, center, radius, color, -1)

                    # 绘制带数字的标注
                    square_side = 10  # 正方形边长
                    top_left = (center[0] - square_side//2, center[1] - square_side//2)
                    bottom_right = (center[0] + square_side//2, center[1] + square_side//2)
                    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)  # 绘制黑色正方形
                    cv2.putText(image, str(gripper_index), (top_left[0] + 2, bottom_right[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    gripper_index += 1  # 更新数字标注
                
                rgb_w_gripper_path = './test_data/rgb_image_w_gripper.png'
                cv2.imwrite(rgb_w_gripper_path, image)  # 保存修改后的图像

            # Add the point cloud to the visualizer
            if SHOW_3D:
                grippers = gg.to_open3d_geometry_list()
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(ret_dict['point_clouds'].astype(np.float32))
                point_cloud.colors = o3d.utility.Vector3dVector(all_color_sampled.astype(np.float32)/255.0)
                o3d.visualization.draw_geometries([point_cloud, *grippers])


            gg = gg[:1]
            print("grasp width : ", gg.widths)
            print("grasp score : ", gg.scores)
            return gg

if __name__ == '__main__':
    # arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--num_sample', type=int, default=96, help="number of depth levels")
    parser.add_argument('--depth_min', type=float, default=0.2, help="number of levels in the correlation pyramid")
    parser.add_argument('--depth_max', type=float, default=1.5, help="width of the correlation pyramid")
    parser.add_argument('--train_2layer', default=True, help="")

    parser.add_argument('--restore_ckpt',
                        default=f'./checkpoints/raftmvs_2layer.pth',
                        help="restore checkpoint")

    #########################################################################
    parser.add_argument('--checkpoint_path', default='./checkpoints/minkuresunet_epoch10.tar')
    parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
    parser.add_argument('--num_point', type=int, default=55000, help='Point Number [default: 15000]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
    parser.add_argument('--graspness_threshold', type=float, default=0, help='graspness threshold')

    args = parser.parse_args()
    eval = MVSGSNetEval(args)

    # test inputs
    # rgb_path = f'./test_data/00100_0000_color.png'
    # ir1_path = f'./test_data/00100_0000_ir_l.png'
    # ir2_path = f'./test_data/00100_0000_ir_r.png' 
    rgb_path = f'./test_data/2_rgb_image.png'
    ir1_path = f'./test_data/2_ir1_image.png'
    ir2_path = f'./test_data/2_ir2_image.png'

    rgb_w_gripper_path = f'./test_data/2_rgb_image_w_gripper.png'
    gg = eval.infer(rgb_path, ir1_path, ir2_path)

