# Code for TSM adapted from the original TSM repo:
# https://github.com/mit-han-lab/temporal-shift-module

import os
import os.path
import numpy as np
import pandas as pd
from numpy.random import randint
import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame_{:010d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True,
                 dense_sample=False, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.csv_folder = '/home/jay/PycharmProjects/assembly101-action-recognition/TSM-action-recognition/my_data/hand_csv'

        if self.modality == 'RGBDiff':
            # Diff needs one more image to calculate diff
            self.new_length += 1

        self._parse_list()
        self._load_csv()

    def _load_csv(self):
        """
        预加载所有 CSV 文件到字典中，键为视频名称（不含扩展名），值为对应的 DataFrame。
        """
        self.hand_data = {}
        if self.csv_folder is None:
            print("CSV 文件夹路径未提供。")
            return
        for csv_file in os.listdir(self.csv_folder):
            if csv_file.endswith('.csv'):
                video_name = os.path.splitext(csv_file)[0]
                csv_path = os.path.join(self.csv_folder, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    self.hand_data[video_name] = df
                except Exception as e:
                    print(f"加载 CSV 文件失败: {csv_path}, 错误: {e}")

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            # print(os.path.join(directory, self.image_tmpl.format(idx)))
            return [Image.open(
                os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format(idx))).convert('RGB')]
            # return [Image.new('RGB', (456, 256), (73, 109, 137))]
        elif self.modality == 'Flow':
            x_img = Image.open(
                os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(
                os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        # 初始化手部位置信息列表
        hands_info = []

        # 获取视频名称（不含扩展名）
        video_name = os.path.splitext(os.path.basename(record.path))[0]
        # 获取对应的 DataFrame
        hand_df = self.hand_data.get(video_name, pd.DataFrame())

        for seg_ind in indices:
            frame_num = int(seg_ind)
            # 查找对应帧的手部位置
            row = hand_df[hand_df['frame_number'] == frame_num]
            if not row.empty:
                left_x = row['left_x'].values[0]
                left_y = row['left_y'].values[0]
                right_x = row['right_x'].values[0]
                right_y = row['right_y'].values[0]
            else:
                # 如果未检测到手部，填充为0.0
                left_x, left_y, right_x, right_y = 0.0, 0.0, 0.0, 0.0
            hands_info.append([left_x, left_y, right_x, right_y])

        # 将手部位置信息转换为张量
        hands_info = torch.tensor(hands_info).float()  # [num_segments, 4]

        process_data = self.transform(images)
        return process_data, record.label, hands_info

    def __len__(self):
        return len(self.video_list)
