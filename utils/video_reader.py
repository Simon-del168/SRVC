import cv2
import numpy as np
//提取指定数量的帧
def fetch_chunk_frames(cap, n_chunk_frames, step_frame, size=None):
    frames = []  #存储提取的帧
    # 判断视频捕获对象类型（OpenCV原生格式 或 YUV格式）
    cap_type = 'cv2_cap' if isinstance(cap, cv2.VideoCapture) else 'yuv_cap'
    # 循环次数：总帧数 = 需要提取的帧数 × 步长（跳过的帧数）
    for f in range(n_chunk_frames * step_frame):
        assert cap.isOpened()
        # 根据步长决定是否实际读取帧内容
        if f % step_frame == 0:    # 当f是步长的整数倍时（目标帧）
            ret, frame = cap.read()   # 读取完整帧（返回是否成功ret和帧数据frame）
        else:  # 非目标帧时（跳过的帧）
            # 如果是YUV格式，调用read_raw读取原始数据（不转换颜色）；否则用OpenCV读取
            ret, _ = cap.read_raw() if cap_type == 'yuv_cap' else cap.read()

        if not ret:
            break
         # 仅处理目标帧（步长整数倍的帧）
        if f % step_frame == 0:
            print(f)   # 打印当前帧序号（调试用）
             # 颜色空间转换：OpenCV默认读取为BGR格式，转换为RGB（更符合通用图像格式）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if size is not None:
                frame = cv2.resize(frame, size)
            frames.append(frame)   # 将处理后的帧存入列表
    return frames  # 返回提取的所有帧
//根据输入的参数 args 配置视频处理的相关参数
//据用户输入的参数（如视频路径、尺寸、采样间隔等），
//计算视频处理所需的所有关键参数（如帧率、采样步长、总帧数等），为后续视频读取和处理提供统一配置。
def get_config(args):
    start_time = args.start_time
    end_time = args.end_time

    hr_path = args.hr_video
    lr_path = args.lr_video

    # 解析HR视频尺寸（参数格式为"宽,高"）
    hr_size = args.hr_size.split(',')    # 分割字符串为列表
    hr_size = (int(hr_size[0]), int(hr_size[1]))   # 转换为整数元组（宽, 高）
    # 解析LR视频尺寸（若未指定则默认HR的1/4）
    if args.lr_size is not None:
        lr_size = args.lr_size.split(',')
        lr_size = (int(lr_size[0]), int(lr_size[1]))
    else:
        lr_size = (hr_size[0] // 4, hr_size[1] // 4)

    # 获取视频格式
    hr_vid_format = hr_path.split('.')[-1]
    lr_vid_format = lr_path.split('.')[-1]
    
     # 初始化HR视频捕获对象（YUV格式需用自定义类，其他用OpenCV）
    hr_cap = VideoCaptureYUV(hr_path, hr_size) if hr_vid_format == 'yuv' else cv2.VideoCapture(hr_path)

    # 计算视频帧率（FPS）
    if hr_vid_format == 'yuv':   # YUV格式无内置FPS信息，从参数中获取
        fps = args.fps   # 参数中的FPS（可能为分数形式，如"30000/1001"）
        if len(fps.split('/')) == 2:  # 处理分数格式
            fps = float(fps.split('/')[0]) / float(fps.split('/')[1])
            fps = int(round(fps))
        else:     # 直接转换为整数
            fps = int(round(float(fps)))
    else:    # 非YUV格式（如MP4），从视频元数据中读取FPS
        fps = int(round(hr_cap.get(cv2.CAP_PROP_FPS)))
    hr_cap.release()  # 释放HR视频捕获对象

    step_time = args.sampling_interval  # 采样间隔（秒）
    step_frame = max(1, int(step_time * fps))   # 采样步长（帧）= 采样间隔 × FPS（至少1帧）
    n_chunk_frames = int(args.update_interval * fps / float(step_frame))   # 每块帧数 = 更新间隔 × FPS / 步长
    # 计算边界阈值（控制迭代次数）
    boundary_threshold = ((n_chunk_frames * args.num_epochs) // args.batch_size) * args.batch_size    
    if args.inference:  # 推理模式时简化为每块帧数
        boundary_threshold = n_chunk_frames

    total_chunks = int((args.end_time - args.start_time) / args.update_interval)   # 总块数 = 总时间 / 更新间隔
    warmup_epochs = 0 if args.coord_frac == 1.0 else 1   # 预热轮数（协调分数为1时不需要）
    num_epochs = args.num_epochs - warmup_epochs    # 有效轮数 = 总轮数 - 预热轮数
    n_chunk_iterations = (n_chunk_frames * num_epochs) // args.batch_size   # 每块迭代次数
    num_frames = total_chunks * n_chunk_frames     # 总帧数 = 总块数 × 每块帧数

    config = {'n_chunk_frames': n_chunk_frames,
              'hr_path': hr_path,
              'hr_size': hr_size,
              'lr_path': lr_path,
              'lr_size': lr_size,
              'fps': fps,
              'boundary_threshold': boundary_threshold,
              'step_frame': step_frame,
              'hr_vid_format': hr_vid_format,
              'lr_vid_format': lr_vid_format,
              'warmup_epochs': warmup_epochs,
              'num_epochs': num_epochs,
              'total_chunks': total_chunks,
              'n_chunk_iterations': n_chunk_iterations,
              'num_frames': num_frames,
              'start_time': start_time,
              'end_time': end_time, }
    return config


m = np.array([[1.0, 1.0, 1.0],
              [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
              [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])
m = m[..., [2, 1, 0]]

//将 YUV 格式的图像转换为 BGR 格式的图像
def YUV2BGR(yuv):
    h = int(yuv.shape[0] / 1.5)
    w = yuv.shape[1]
    y = yuv[:h]
    h_u = h // 4
    h_v = h // 4
    u = yuv[h:h + h_u]
    v = yuv[-h_v:]
    u = np.reshape(u, (h_u * 2, w // 2))
    v = np.reshape(v, (h_v * 2, w // 2))
    u = cv2.resize(u, (w, h), interpolation=cv2.INTER_CUBIC)
    v = cv2.resize(v, (w, h), interpolation=cv2.INTER_CUBIC)
    yuv = np.concatenate([y[..., None], u[..., None], v[..., None]], axis=-1)

    bgr = np.dot(yuv, m)
    bgr[:, :, 2] -= 179.45477266423404
    bgr[:, :, 1] += 135.45870971679688
    bgr[:, :, 0] -= 226.8183044444304
    bgr = np.clip(bgr, 0, 255)

    return bgr.astype(np.uint8)

//针对 YUV 格式视频（无封装格式的原始像素数据），实现类似 OpenCV VideoCapture 的功能，支持读取原始数据并转换为 BGR 图像。
class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.width, self.height = size
        self.frame_len = (self.width * self.height * 3) // 2
        self.f = open(filename, 'rb')
        self.is_opened = True
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            self.is_opened = False
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, None
        bgr = YUV2BGR(yuv)
        return ret, bgr

    def isOpened(self):
        return self.is_opened

    def release(self):
        try:
            self.f.close()
        except Exception as e:
            print(str(e))
