import math
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class DualRealSenseModule:
    def __init__(self, real_time_view=False, rgb_size=[640, 480]):
        self.state = AppState()
        
        # 创建两个pipeline实例
        self.pipeline1 = rs.pipeline()
        self.pipeline2 = rs.pipeline()
        
        # 配置两个相机
        self.config1 = rs.config()
        self.config2 = rs.config()
        
        # 获取所有连接的设备
        self.ctx = rs.context()
        self.devices = list(self.ctx.query_devices())
        
        if len(self.devices) < 2:
            raise RuntimeError("需要连接两个RealSense相机")
            
        # 为每个相机指定序列号
        self.serial_1 = self.devices[0].get_info(rs.camera_info.serial_number)
        self.serial_2 = self.devices[1].get_info(rs.camera_info.serial_number)
        
        # 分别配置两个相机
        self.config1.enable_device(self.serial_1)
        self.config2.enable_device(self.serial_2)
        
        # 配置流
        for config in [self.config1, self.config2]:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, rgb_size[0], rgb_size[1], rs.format.bgr8, 30)
        
        # 启动流
        self.pipeline1.start(self.config1)
        self.pipeline2.start(self.config2)
        
        # 获取相机参数
        self.profile1 = self.pipeline1.get_active_profile()
        self.profile2 = self.pipeline2.get_active_profile()
        
        # 设置对齐器
        self.align = rs.align(rs.stream.color)
        
        # 获取深度比例
        self.depth_scale1 = self.profile1.get_device().first_depth_sensor().get_depth_scale()
        self.depth_scale2 = self.profile2.get_device().first_depth_sensor().get_depth_scale()
        
        if real_time_view:
            cv2.namedWindow(self.state.WIN_NAME + "_1", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.state.WIN_NAME + "_2", cv2.WINDOW_AUTOSIZE)
    
    def get_camera_intrinsics(self, profile):
        color_stream = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_stream.get_intrinsics()
        
        mtx = [intrinsics.width, intrinsics.height, intrinsics.ppx, intrinsics.ppy, intrinsics.fx, intrinsics.fy]
        camIntrinsics = np.array([[mtx[4], 0, mtx[2]],
                                 [0, mtx[5], mtx[3]],
                                 [0, 0, 1.]])
        
        return camIntrinsics, intrinsics.coeffs

    def get_data(self):
        """获取两个相机的数据"""
        step = 0
        while True:
            # 等待两个相机的帧
            frames1 = self.pipeline1.wait_for_frames()
            frames2 = self.pipeline2.wait_for_frames()
            
            # 对齐深度帧和彩色帧
            aligned_frames1 = self.align.process(frames1)
            aligned_frames2 = self.align.process(frames2)
            
            # 获取深度帧和彩色帧
            depth_frame1 = aligned_frames1.get_depth_frame()
            color_frame1 = aligned_frames1.get_color_frame()
            depth_frame2 = aligned_frames2.get_depth_frame()
            color_frame2 = aligned_frames2.get_color_frame()
            
            if not all([depth_frame1, color_frame1, depth_frame2, color_frame2]):
                continue
                
            # step += 1
            # if step < 5:  # 跳过前几帧以等待相机稳定
            #     continue
                
            # 转换为numpy数组
            depth_image1 = np.asanyarray(depth_frame1.get_data())
            color_image1 = np.asanyarray(color_frame1.get_data())
            depth_image2 = np.asanyarray(depth_frame2.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())
            
            # 获取相机参数
            camIntrinsics1, distCoeffs1 = self.get_camera_intrinsics(self.profile1)
            camIntrinsics2, distCoeffs2 = self.get_camera_intrinsics(self.profile2)
            
            break
            
        return (color_image1, depth_image1, camIntrinsics1, distCoeffs1,
                color_image2, depth_image2, camIntrinsics2, distCoeffs2)

def _init_rs_cameras(real_time_view=False):
    rs_module = DualRealSenseModule(real_time_view=real_time_view)

    return rs_module

def save_rgbd_seqs(rs_module, save_path='./force_feedback/replay_data/1/rgbd', saving_freq=10):
    view_step = 0
    
    os.makedirs(save_path, exist_ok=True)

    timesleep = 1. / saving_freq
    
    while True:
        try:
            # 获取两个相机的数据
            (color_image1, depth_image1, camIntrinsics1, distCoeffs1,
             color_image2, depth_image2, camIntrinsics2, distCoeffs2) = rs_module.get_data()
            
            # 保存数据
            for cam_idx, (color_img, depth_img, cam_intrinsics) in enumerate([
                (color_image1, depth_image1, camIntrinsics1),
                (color_image2, depth_image2, camIntrinsics2)
            ]):
                # 保存图像和参数
                np.save(os.path.join(save_path, f'color_image_cam{cam_idx}_{view_step}.npy'), color_img)
                np.save(os.path.join(save_path, f'depth_image_cam{cam_idx}_{view_step}.npy'), depth_img)
                np.save(os.path.join(save_path, f'camIntrinsics_cam{cam_idx}.npy'), cam_intrinsics)
                cv2.imwrite(os.path.join(save_path, f'color_image_cam{cam_idx}_{view_step}.jpg'), color_img)
            
            view_step += 1
            print('view_step: ', view_step)

            time.sleep(timesleep)
                
        except KeyboardInterrupt:
            print("正在退出...")
            break

def get_rgbd(rs_module):
    
    # 获取两个相机的数据
    (color_image1, depth_image1, camIntrinsics1, distCoeffs1,
        color_image2, depth_image2, camIntrinsics2, distCoeffs2) = rs_module.get_data()
    
    return color_image1, depth_image1, camIntrinsics1, color_image2, depth_image2, camIntrinsics2

if __name__ == '__main__':
    cameras = _init_rs_cameras()

    image1, _, _, image2, _, _ = get_rgbd(cameras)

    cv2.imshow('image1', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image2', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save_rgbd_seqs(cameras, './test_sync', 30)