# process_videos.py

import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

class HandPositionExtractor:
    def __init__(self, max_num_hands=2, detection_confidence=0.5, tracking_confidence=0.5, mirror_image=False):
        """
        初始化 MediaPipe 手部检测模型。

        :param max_num_hands: 最多检测的手部数量。
        :param detection_confidence: 手部检测的最小置信度。
        :param tracking_confidence: 手部跟踪的最小置信度。
        :param mirror_image: 是否对图像进行水平翻转（镜像）。
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils  # 用于绘制手部关键点
        self.mirror_image = mirror_image

    def extract_hand_positions(self, image):
        """
        提取图像中的左右手中心位置（手腕位置）。

        :param image: 输入图像（BGR格式）。
        :return: 左右手中心位置列表，每个位置为 [x, y]，归一化到 [0, 1] 范围。
                 如果某只手未检测到，则使用 [0.0, 0.0] 作为填充。
        """
        # 如果需要镜像翻转图像
        if self.mirror_image:
            image = cv2.flip(image, 1)

        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        left_hand = [0.0, 0.0]
        right_hand = [0.0, 0.0]

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' 或 'Right'
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                pos = [wrist.x, wrist.y]
                if label == 'Right':
                    left_hand = pos
                elif label == 'Left':
                    right_hand = pos

        # 如果图像被镜像翻转，交换左右手的位置
        if self.mirror_image:
            left_hand, right_hand = right_hand, left_hand

        return left_hand, right_hand  # [left_x, left_y], [right_x, right_y]

    def draw_hand_positions(self, image, hand_landmarks):
        """
        在图像上绘制手部关键点。

        :param image: 输入图像（BGR格式）。
        :param hand_landmarks: 手部关键点。
        :return: 绘制后的图像。
        """
        self.mp_drawing.draw_landmarks(
            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

def process_videos(input_folder, output_folder, annotated_folder, mirror_image=False):
    """
    处理文件夹中的所有视频，检测手部位置并保存到 CSV 文件和注释视频中。

    :param input_folder: 输入视频文件夹路径。
    :param output_folder: 输出 CSV 文件夹路径。
    :param annotated_folder: 输出注释视频文件夹路径。
    :param mirror_image: 是否对图像进行水平翻转（镜像）。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    # 支持的视频文件扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print("在指定的文件夹中未找到视频文件。")
        return

    # 初始化手部检测器
    hand_extractor = HandPositionExtractor(mirror_image=mirror_image)

    # 遍历所有视频文件
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 准备 CSV 文件
        csv_filename = os.path.splitext(video_file)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_filename)
        data = {
            'frame_number': [],
            'left_x': [],
            'left_y': [],
            'right_x': [],
            'right_y': []
        }

        # 准备注释视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编解码器
        annotated_video_path = os.path.join(annotated_folder, os.path.splitext(video_file)[0] + "_annotated.mp4")
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        # 逐帧处理
        for frame_num in tqdm(range(frame_count), desc=f"Processing {video_file}", leave=False):
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取帧 {frame_num} 从视频 {video_file}")
                # 记录为未检测到手部
                data['frame_number'].append(frame_num)
                data['left_x'].append(0.0)
                data['left_y'].append(0.0)
                data['right_x'].append(0.0)
                data['right_y'].append(0.0)
                # 写入未标注的帧
                out.write(frame)
                continue

            # 提取手部位置
            left_hand, right_hand = hand_extractor.extract_hand_positions(frame)

            # 记录数据
            data['frame_number'].append(frame_num)
            data['left_x'].append(left_hand[0])
            data['left_y'].append(left_hand[1])
            data['right_x'].append(right_hand[0])
            data['right_y'].append(right_hand[1])

            # 绘制手部位置
            h, w, _ = frame.shape
            if left_hand != [0.0, 0.0]:
                left_x, left_y = int(left_hand[0] * w), int(left_hand[1] * h)
                cv2.circle(frame, (left_x, left_y), 10, (0, 255, 0), -1)  # 绿色圆点表示左手
                cv2.putText(frame, 'Left Hand', (left_x + 10, left_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if right_hand != [0.0, 0.0]:
                right_x, right_y = int(right_hand[0] * w), int(right_hand[1] * h)
                cv2.circle(frame, (right_x, right_y), 10, (255, 0, 0), -1)  # 蓝色圆点表示右手
                cv2.putText(frame, 'Right Hand', (right_x + 10, right_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 将注释后的帧写入输出视频
            out.write(frame)

        # 释放视频捕捉和写入
        cap.release()
        out.release()

        # 保存到 CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"已保存手部位置信息到 {csv_path}")
        print(f"已保存注释视频到 {annotated_video_path}")

if __name__ == '__main__':
    # 配置输入和输出文件夹路径
    input_folder = '/media/jay/hard_disk/data0904/video/'          # 替换为您的视频文件夹路径
    output_folder = '../my_data/hand_csv'     # 替换为您希望保存 CSV 的文件夹路径
    annotated_folder = '../my_data/hand_csv'  # 替换为您希望保存注释视频的文件夹路径

    # 运行处理
    process_videos(input_folder, output_folder, annotated_folder)
