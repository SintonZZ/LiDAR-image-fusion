import numpy as np
import laspy
import cv2
from pyproj import Transformer, CRS
import os

# ==========================================
# 1. 参数配置区域 (请根据实际情况修改路径)
# ==========================================

# 文件路径
LAS_FILE_PATH = "20220728_example.laz"  # 您的点云文件路径
IMG_FILE_PATH = "full.jpg"        # 您的遥感影像路径
OUTPUT_PATH   = "fusion_result.jpg" # 结果保存路径

# 坐标系定义
# 源坐标系: 点云 (WGS84 UTM Zone 50N) -> EPSG:32650
SRC_EPSG = 32650 
# 目标坐标系: 影像 (CGCS2000 3-degree Zone 40) -> EPSG:4549
DST_EPSG = 4549

# 相机内参 (Intrinsics)
F_MM = 70.5             # 焦距 (mm)
PIXEL_SIZE = 0.006      # 像元大小 (mm)
SENSOR_CX_OFFSET = 0.0  # 主点 x 偏移 (mm)
SENSOR_CY_OFFSET = 0.12 # 主点 y 偏移 (mm)

# 影像尺寸 (Pixels)
IMG_WIDTH  = 11310
IMG_HEIGHT = 17310

# 相机外参 (Extrinsics) - World System (CGCS2000 Zone 40)
# 相机中心 T (X, Y, Z)
CAMERA_POS = np.array([396163.276250, 3552863.486680, 3718.316940])

# 旋转矩阵 R (World -> Camera)
CAMERA_ROT = np.array([
    [ 0.999906394, 0.011688825, 0.007111621],
    [-0.011621217, 0.999887581,-0.009474858],
    [-0.007221571, 0.009391326, 0.999929823]
])

# ==========================================
# 2. 核心处理函数
# ==========================================

def run_fusion():
    print(f"--- 开始处理 ---")
    
    # -------------------------------------------------
    # 步骤 1: 读取影像
    # -------------------------------------------------
    if not os.path.exists(IMG_FILE_PATH):
        print(f"错误: 找不到影像文件 {IMG_FILE_PATH}")
        return
    
    print(f"正在读取影像: {IMG_FILE_PATH} ...")
    # 注意: opencv 读取超大图可能会慢，且如果图太大可能内存溢出
    # 如果内存不够，需要使用 resize 或切片处理，这里默认直接读取
    img = cv2.imread(IMG_FILE_PATH)
    if img is None:
        print("错误: 影像读取失败。")
        return

    # -------------------------------------------------
    # 步骤 2: 读取 .laz 点云
    # -------------------------------------------------
    print(f"正在读取点云: {LAS_FILE_PATH} ...")
    try:
        las = laspy.read(LAS_FILE_PATH)
    except Exception as e:
        print(f"读取 LAZ 失败 (是否安装了 lazrs?): {e}")
        return

    # 获取原始坐标
    x_source = las.x
    y_source = las.y
    z_source = las.z
    num_points = len(x_source)
    print(f"点云数量: {num_points}")

    # -------------------------------------------------
    # 步骤 3: 坐标投影转换 (UTM 50N -> CGCS2000 Zone 40)
    # -------------------------------------------------
    print("正在进行坐标转换 (UTM -> CGCS2000) ...")
    transformer = Transformer.from_crs(SRC_EPSG, DST_EPSG, always_xy=True)
    
    # 执行转换
    x_world, y_world = transformer.transform(x_source, y_source)
    # 假设 Z 值不需要大地水准面变换，直接沿用
    z_world = z_source

    # 验证转换结果是否合理 (检查第一个点)
    print(f"  [验证] 原始 X: {x_source[0]:.2f} -> 转换后 X: {x_world[0]:.2f}")
    print(f"  [验证] 相机 X: {CAMERA_POS[0]:.2f}")
    if abs(x_world[0] - CAMERA_POS[0]) > 5000:
        print("⚠️ 警告: 转换后的点云与相机距离依然过远 (>5km)，请检查 EPSG 代码！")

    # -------------------------------------------------
    # 步骤 4: 3D -> 2D 投影计算 (向量化加速版)
    # -------------------------------------------------
    print("正在进行投影计算 ...")

    # 4.1 转换为 Numpy 矩阵 (N, 3)
    points_world = np.vstack((x_world, y_world, z_world)).T
    
    # 4.2 计算相对于相机的位移 (P_world - T)
    # 这一步将坐标系原点移到了相机中心
    points_local = points_world - CAMERA_POS
    
    # 4.3 旋转到相机坐标系 (P_cam = R * P_local)
    # 矩阵乘法: (N, 3) dot (3, 3).T
    points_cam = points_local @ CAMERA_ROT.T
    
    # 提取相机系坐标
    Xc = points_cam[:, 0]
    Yc = points_cam[:, 1]
    Zc = points_cam[:, 2]

    # 4.4 深度处理与剔除
    # 在航测中，相机向下看，若 Z 轴向上，则 Zc 为负值；若 Z 轴向下，则 Zc 为正值。
    # 我们这里取深度的绝对值进行计算
    depth = -Zc # 假设 Zc 是负数 (相机在 3700m，点在 30m，差值 -3670)
    
    # 剔除相机背后的点 (深度 <= 0)
    valid_mask = depth > 0
    
    # 4.5 物理成像平面坐标 (mm)
    # x = f * X/Z
    x_film = F_MM * (Xc / depth)
    y_film = F_MM * (Yc / depth)

    # 4.6 像素坐标转换
    # 图像中心
    cx_img = IMG_WIDTH / 2.0
    cy_img = IMG_HEIGHT / 2.0
    
    # u = cx + (x_film - offset_x) / pixel_size
    u = cx_img + (x_film - SENSOR_CX_OFFSET) / PIXEL_SIZE
    
    # v = cy - (y_film - offset_y) / pixel_size 
    # (注意: 图像坐标系 Y 向下，物理坐标系通常 Y 向上，故用减号)
    v = cy_img - (y_film - SENSOR_CY_OFFSET) / PIXEL_SIZE

    # -------------------------------------------------
    # 步骤 5: 筛选与图像绘制
    # -------------------------------------------------
    print("正在渲染融合图像 ...")
    
    # 5.1 更新 Mask: 只保留在图像范围内的点
    valid_mask = valid_mask & (u >= 0) & (u < IMG_WIDTH) & \
                              (v >= 0) & (v < IMG_HEIGHT)
    
    valid_u = u[valid_mask].astype(int)
    valid_v = v[valid_mask].astype(int)
    valid_depth = depth[valid_mask]
    
    num_valid = len(valid_u)
    print(f"投影在图像范围内的点数: {num_valid}")
    
    if num_valid == 0:
        print("没有点落在图像上，请检查坐标系或参数。")
        return

    # 5.2 颜色映射 (Color Mapping) - 根据深度上色，越近越红，越远越蓝
    # 归一化深度到 0-255
    d_min, d_max = np.min(valid_depth), np.max(valid_depth)
    if d_max - d_min > 0:
        norm_depth = 255 * (valid_depth - d_min) / (d_max - d_min)
        norm_depth = norm_depth.astype(np.uint8)
        # 使用 JET 色带 (OpenCV)
        colors = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        colors = colors.squeeze() # (N, 3)
    else:
        colors = np.full((num_valid, 3), (0, 255, 0), dtype=np.uint8) # 纯绿

    # 5.3 绘制
    # 为了速度，我们直接操作像素数组，比 cv2.circle 循环快
    # 但为了点稍微大一点，还是用 circle 比较好看
    # 如果点太多(>百万)，建议只画 1个像素
    
    # 创建一个副本以免破坏原图
    canvas = img.copy()
    
    # 这里使用简单的循环，对于几十万个点可能会慢几秒
    # 若需极速，可直接赋值 canvas[valid_v, valid_u] = colors
    for i in range(num_valid):
        # 颜色: cv2 是 BGR 顺序
        color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        # 半径: 3像素
        cv2.circle(canvas, (valid_u[i], valid_v[i]), 3, color, -1)

    # -------------------------------------------------
    # 步骤 6: 保存结果
    # -------------------------------------------------
    cv2.imwrite(OUTPUT_PATH, canvas)
    print(f"✅ 处理完成！结果已保存至: {OUTPUT_PATH}")

# 运行主程序
if __name__ == "__main__":
    run_fusion()
