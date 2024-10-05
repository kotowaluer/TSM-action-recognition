import os
import cv2

# 定义文件夹路径
input_folder = '/media/jay/TOSHIBA/assembly101'  # 源文件夹路径
output_folder = '/media/jay/hard_disk/assembly101-download-scripts/Assembly101/frames'  # 目标文件夹路径

# 遍历所有子文件夹
for subdir in os.listdir(input_folder):
    print('current:{}'.format(subdir))
    subdir_path = os.path.join(input_folder, subdir)
    if os.path.isdir(subdir_path):
        video_files = [f for f in os.listdir(subdir_path) if f.endswith(('.mp4', '.avi', '.mkv'))]

        # 假设每个子文件夹中只有一个视频文件
        if video_files:
            video_path = os.path.join(subdir_path, video_files[0])
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            # 获取视频文件的名称（不带扩展名）
            video_name = os.path.splitext(video_files[0])[0]

            # 在目标文件夹中创建对应的子文件夹
            output_subdir = os.path.join(output_folder, subdir, video_name)
            os.makedirs(output_subdir, exist_ok=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 构建帧的文件名
                frame_filename = os.path.join(output_subdir, f'{video_name}_{frame_count:010d}.jpg')

                # 保存帧
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(frame_filename, frame)

                frame_count += 1

            cap.release()

print("帧提取完成！")
