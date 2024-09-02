import numpy as np

class CameraMgr():
    def __init__(self):

        ## 内参 K (3,3)
        # 定义RGB相机的内部参数矩阵。这个矩阵包括焦距和光心坐标，用于将3D点投影到2D图像平面
        self.intrinsic_rgb = np.array([[910.813728, 0, 640],
                              [0, 910.813728, 360],
                              [0, 0, 1]])
        # 定义红外(IR)相机的内部参数矩阵。与RGB相机类似，但参数可能略有不同，反映了不同的焦距或光心位置
        self.intrinsic_ir = np.array([[896.866, 0, 640],
                             [0, 896.866, 360],
                             [0, 0, 1]])
        
        ## 外参 E=[R|t] (4,4) / 这里最后只用 (3, 4)
        # 初始化RGB相机的姿态矩阵为4x4的单位矩阵。这表示相机位于原点，朝向Z轴正方向，没有任何旋转或平移
        self.pose_rgb = np.eye(4, dtype=np.float32)
        # 定义第一个IR相机的姿态矩阵。这里只有X轴上有一个小的平移(-0.0151米)，其他方向无旋转和平移
        self.pose_ir1 = np.array([[1., 0.0, 0.0, -0.0151],
                             [0.0, 1., 0.0, 0.0],
                             [0.0, 0.0, 1., 0.0],
                             [0, 0, 0, 1]])
        # 定义第二个IR相机的姿态矩阵。与第一个IR相机类似，但X轴上的平移更大(-0.0701米)
        self.pose_ir2 = np.array([[1., 0.0, 0.0, -0.0701],
                             [0.0, 1., 0.0, 0.0],
                             [0.0, 0.0, 1., 0.0],
                             [0, 0, 0, 1]])
        ## 投影矩阵 P = K E = (3, 3) (3, 4) = (3, 4) 
        # P2 = P P3 = (3, 4)(4, 1) = (3, 1) -> (2, 1)

    # 模拟D415 RGB相机的内部参数
    def sim_d415_rgb(self):
        intrinsic = self.intrinsic_rgb.copy()
        # 将焦距和光心坐标缩小到原来的一半，用于模拟不同的分辨率或视场。
        intrinsic[: 2] *= 0.5
        # 将光心坐标进一步偏移，通常用于校准或调整光心位置
        intrinsic[0:2, 2] -= 0.5
        return intrinsic

    def sim_d415_ir(self):
        intrinsic = self.intrinsic_ir.copy()
        # 将焦距和光心坐标缩小到原来的一半
        intrinsic[: 2] *= 0.5
        # 将光心坐标进一步偏移
        intrinsic[0:2, 2] -= 0.5
        return intrinsic

    # IR相机到RGB相机的相对姿态
    def sim_ir12rgb(self):
        # 返回第一个IR相机姿态的逆矩阵，这表示从IR相机坐标系转换到RGB相机坐标系（这里假设RGB相机处于原点）
        return np.linalg.inv(self.pose_ir1)

    def sim_d415_rgb_pose(self):
        return self.pose_rgb.copy()

    def sim_d415_ir1_pose(self):
        return self.pose_ir1.copy()

    def sim_d415_ir2_pose(self):
        return self.pose_ir2.copy()

