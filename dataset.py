import torch
import torch.utils.data as data
import numpy as np
import os
import trimesh


class ModelNet10Dataset(data.Dataset):
    def __init__(self, root_dir, split='train', num_points=1024):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.data_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name, split)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.off'):
                        self.data_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, idx):
        path = self.data_paths[idx]
        label = self.labels[idx]

        mesh = trimesh.load(path, force='mesh')

        # Sample points from the mesh surface
        points = mesh.sample(self.num_points)

        # Normalize: center and scale to unit sphere
        points = points - np.mean(points, axis=0)
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / dist

        points = torch.from_numpy(points).float().transpose(1, 0)
        label = torch.tensor(label, dtype=torch.long)

        return points, label
