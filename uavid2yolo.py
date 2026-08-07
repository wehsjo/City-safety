import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ====================== 配置参数 ======================
# 你要提取的目标类别（此处以 'static car' 为例，ID=3）
TARGET_CLASS_ID = 3  # 0:clutter, 1:building, 2:road, 3:static car, 4:tree, 5:vegetation, 6:human, 7:moving car

# 类别对应的RGB颜色 (用于从标签图中筛选)
CLASS_RGB_MAP = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (128, 64, 128),
    3: (192, 0, 192),
    4: (0, 128, 0),
    5: (128, 128, 0),
    6: (64, 64, 0),
    7: (64, 0, 128),
}
TARGET_RGB = CLASS_RGB_MAP[TARGET_CLASS_ID]

# 数据路径
DATA_ROOT = "./dataset"          # UAVid原始数据集根目录
OUTPUT_ROOT = "./yolo_dataset"   # 输出YOLO数据集的根目录

# ====================== 重命名开关 ======================
RENAME_IMAGES = True              # True: 重命名为 seqX_000000.png; False: 保持原名
COPY_IMAGES = True               # True: 复制图片; False: 创建软链接（节省空间）


# ====================== 核心转换函数 ======================
def convert_mask_to_yolo(mask_path, img_shape, target_rgb, class_id, min_area=50):
    """
    从语义分割掩码中提取目标类别的边界框，转为YOLO格式
    返回: list of [class_id, x_center, y_center, width, height] (归一化)
    """
    # 读取掩码 (BGR格式)
    mask = cv2.imread(mask_path)
    if mask is None:
        return []
    
    # 生成二值掩码：像素颜色等于目标RGB的置为255，其余为0
    target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])  # RGB -> BGR
    lower = np.array(target_bgr, dtype=np.uint8)
    upper = np.array(target_bgr, dtype=np.uint8)
    binary_mask = cv2.inRange(mask, lower, upper)
    
    # 寻找连通区域的外轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img_shape[:2]
    yolo_annos = []
    
    for cnt in contours:
        # 过滤掉过小的噪声区域
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 计算最小外接矩形 (轴对齐)
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # 转换为YOLO归一化坐标
        x_center = (x + bw / 2.0) / w
        y_center = (y + bh / 2.0) / h
        width = bw / w
        height = bh / h
        
        # 坐标截断防止越界
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width = min(width, 1.0)
        height = min(height, 1.0)
        
        yolo_annos.append([class_id, x_center, y_center, width, height])
    
    return yolo_annos


def get_new_filename(seq_name, original_filename, rename_flag):
    """
    根据重命名开关生成新文件名
    """
    if rename_flag:
        # 提取原始文件名（不含扩展名），例如 "000000"
        name_without_ext = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1]
        return f"{seq_name}_{name_without_ext}{ext}"
    else:
        return original_filename


# ====================== 主处理流程 ======================
def process_split(split_name):
    """
    处理 train / val / test 中的一个子集
    """
    src_split_dir = os.path.join(DATA_ROOT, f"uavid_{split_name}")
    if not os.path.exists(src_split_dir):
        print(f"警告: {src_split_dir} 不存在，跳过")
        return
    
    # 获取所有序列文件夹 (如 seq1, seq2, ...)
    seq_dirs = [d for d in os.listdir(src_split_dir) 
                if os.path.isdir(os.path.join(src_split_dir, d))]
    
    # 用于统计
    total_images = 0
    total_objects = 0
    
    for seq in tqdm(seq_dirs, desc=f"处理 {split_name}"):
        seq_path = os.path.join(src_split_dir, seq)
        img_dir = os.path.join(seq_path, "images")
        label_dir = os.path.join(seq_path, "labele")  # 注意: UAVid文件夹名是 'labele'
        
        # 对于 test 集，没有 label 文件夹，直接跳过
        if not os.path.exists(label_dir):
            continue
        
        # 获取该序列下所有图片
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, img_file)  # 标签与图片同名
            
            # 读取图片获取尺寸
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}，跳过")
                continue
            img_shape = img.shape
            
            # 执行转换，获得YOLO标注
            yolo_annos = convert_mask_to_yolo(label_path, img_shape, TARGET_RGB, TARGET_CLASS_ID)
            
            # 如果该图没有目标对象，则跳过
            if len(yolo_annos) == 0:
                continue
            
            total_images += 1
            total_objects += len(yolo_annos)
            
            # 生成新文件名（如果需要）
            new_filename = get_new_filename(seq, img_file, RENAME_IMAGES)
            
            # 确定输出路径
            out_img_dir = os.path.join(OUTPUT_ROOT, "images", split_name)
            out_label_dir = os.path.join(OUTPUT_ROOT, "labels", split_name)
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_label_dir, exist_ok=True)
            
            # 复制或软链接图片到输出目录
            out_img_path = os.path.join(out_img_dir, new_filename)
            if COPY_IMAGES:
                cv2.imwrite(out_img_path, img)
            else:
                # 创建软链接（节省磁盘空间）
                if os.path.exists(out_img_path):
                    os.remove(out_img_path)
                os.symlink(os.path.abspath(img_path), out_img_path)
            
            # 写入YOLO标注文件
            label_filename = new_filename.replace('.png', '.txt')
            out_label_path = os.path.join(out_label_dir, label_filename)
            with open(out_label_path, 'w') as f:
                for anno in yolo_annos:
                    f.write(f"{anno[0]} {anno[1]:.6f} {anno[2]:.6f} {anno[3]:.6f} {anno[4]:.6f}\n")
    
    print(f"  {split_name} 集: 共处理 {total_images} 张图片, 检测到 {total_objects} 个目标对象")


# ====================== 生成 data.yaml ======================
def create_data_yaml():
    """生成YOLO数据集配置文件"""
    yaml_content = f"""# UAVid 转 YOLO 数据集配置
# 目标类别: {TARGET_CLASS_ID} - {list(CLASS_RGB_MAP.keys()).index(TARGET_CLASS_ID) if TARGET_CLASS_ID in CLASS_RGB_MAP else 'unknown'}

path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test





nc: 1
names: ['{list(CLASS_RGB_MAP.values())[TARGET_CLASS_ID]}']
"""
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"已生成配置文件: {yaml_path}")


# ====================== 运行转换 ======================
if __name__ == "__main__":
    print("=" * 60)
    print("UAVid -> YOLO 格式转换器")
    print(f"目标类别 ID: {TARGET_CLASS_ID}")
    print(f"重命名开关: {'开启' if RENAME_IMAGES else '关闭'}")
    print(f"复制模式: {'复制图片' if COPY_IMAGES else '创建软链接'}")
    print("=" * 60)
    
    # 依次处理 train, val, test
    for split in ["train", "val", "test"]:
        process_split(split)
    
    # 生成配置文件
    create_data_yaml()
    
    print("\n✅ 转换完成！")
    print(f"输出目录: {OUTPUT_ROOT}")