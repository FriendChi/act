import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        """
        初始化函数，设置episode IDs、数据集目录、相机名称列表以及归一化统计信息。
        """
        super(EpisodicDataset, self).__init__()
        self.episode_ids = episode_ids  # 每个episode的ID列表
        self.dataset_dir = dataset_dir  # 数据集所在的目录路径
        self.camera_names = camera_names  # 使用的相机名称列表
        self.norm_stats = norm_stats  # 包含'action', 'qpos'等的归一化统计信息
        self.is_sim = None  # 标记是否为仿真环境
        self.__getitem__(0)  # 通过获取第1个item初始化self.is_sim

    def __len__(self):
        """
        返回数据集中episode的数量。
        """
        return len(self.episode_ids)

    def __getitem__(self, index):
        """
        根据index返回指定episode的数据，包括图像、位置、速度和动作等信息。
        """
        sample_full_episode = False  # 是否采样整个episode，默认不采样

        episode_id = self.episode_ids[index]  # 获取当前episode ID
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')  # 构造文件路径
        with h5py.File(dataset_path, 'r') as root:  # 打开HDF5文件
            is_sim = root.attrs['sim']  # 判断是否为仿真环境
            original_action_shape = root['/action'].shape  # 动作的原始形状
            episode_len = original_action_shape[0]  # 当前episode长度（时间步长数）
            if sample_full_episode:
                start_ts = 0  # 如果是采样整个episode，则从头开始
            else:
                start_ts = np.random.choice(episode_len)  # 随机选择一个时间点作为起始点
            
            # 获取选定时间点的观察数据
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]  # 对每个相机，获取其图像
            
            # 获取从start_ts开始的所有动作
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # 特殊情况处理，使时间步对齐
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim  # 设置是否为仿真环境标记
        padded_action = np.zeros(original_action_shape, dtype=np.float32)  # 创建与原始动作相同形状的填充数组
        padded_action[:action_len] = action  # 填充实际的动作数据
        is_pad = np.zeros(episode_len)  # 创建填充标志数组
        is_pad[action_len:] = 1  # 超过实际动作长度的部分标记为填充

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])  # 收集所有相机的图像
        all_cam_images = np.stack(all_cam_images, axis=0)  # 将图像堆叠成一个numpy数组

        # 构建返回的张量
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 图像通道调整顺序
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # 归一化图像并转换数据类型为float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad  # 返回包含图像、位置、动作和填充标志的数据


def get_norm_stats(dataset_dir, num_episodes):
    """
    计算并返回给定目录下指定数量的episode文件中的动作(action)和位置(qpos)数据的归一化统计信息。
    
    参数:
    - dataset_dir: 数据集所在的目录路径。
    - num_episodes: 要考虑的episode数量。
    
    返回:
    - stats: 包含动作和位置数据的归一化均值和标准差的字典。
    """
    all_qpos_data = []  # 存储所有episodes的位置数据
    all_action_data = []  # 存储所有episodes的动作数据
    
    for episode_idx in range(num_episodes):
        # 构造每个episode对应的HDF5文件路径
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        
        # 打开HDF5文件并读取数据
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]  # 获取位置数据
            qvel = root['/observations/qvel'][()]  # 获取速度数据，但未使用
            action = root['/action'][()]  # 获取动作数据
            
        # 将numpy数组转换为PyTorch张量，并添加到对应列表中
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    
    # 将列表中的所有张量堆叠成一个大张量
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    
    # 计算动作数据的归一化参数
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)  # 动作数据的均值
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)  # 动作数据的标准差
    action_std = torch.clip(action_std, 1e-2, np.inf)  # 对标准差进行裁剪，避免过小值导致数值不稳定
    
    # 计算位置数据的归一化参数
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)  # 位置数据的均值
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)  # 位置数据的标准差
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # 同样对位置数据的标准差进行裁剪
    
    # 创建并返回包含归一化统计信息的字典
    stats = {
        "action_mean": action_mean.numpy().squeeze(),  # 动作数据的均值
        "action_std": action_std.numpy().squeeze(),  # 动作数据的标准差
        "qpos_mean": qpos_mean.numpy().squeeze(),  # 位置数据的均值
        "qpos_std": qpos_std.numpy().squeeze(),  # 位置数据的标准差
        "example_qpos": qpos  # 示例位置数据（最后一个episode的位置数据）
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
