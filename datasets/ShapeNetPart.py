import os
import json
import numpy as np
import torch
import torch.utils.data as data
from typing import List, Dict, Tuple, Optional, Set

class ShapeNetPart(data.Dataset):
    """
    ShapeNet Part Segmentation Dataset (v0_normal)
    - Expects structure:
        root/
          synsetoffset2category.txt
          train_test_split/
            shuffled_train_file_list.json
            shuffled_test_file_list.json
          02691156/xxxxxxxxxxxxxxxxxxxx.txt
          02958343/xxxxxxxxxxxxxxxxxxxx.txt
          ...
    - Each .txt line: x y z nx ny nz part_id
      (if a parallel .seg exists, it is also supported)
    """

    # Cache global label set & remap across all instances to keep num_parts consistent
    _GLOBAL_LABEL_SET: Optional[Set[int]] = None
    _GLOBAL_REMAP: Optional[Dict[int, int]] = None
    _GLOBAL_NUM_PARTS: Optional[int] = None

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 num_points: int = 1024,
                 category: Optional[str] = None,
                 normal: bool = False):
        self.root = os.path.abspath(root)
        self.split = split.lower()
        if self.split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test', got: {split}")
        self.num_points = int(num_points)
        self.normal = bool(normal)

        # Load category mapping (name -> synset); we will use synset IDs as category keys
        self.name2synset, self.synset_list = self._load_categories()

        # Optional filter to a single category (either name like 'Car' or synset like '02958343')
        if category is not None:
            if category in self.name2synset:
                self.filter_synsets = {self.name2synset[category]}
            else:
                self.filter_synsets = {category}
        else:
            self.filter_synsets = set(self.synset_list)

        # Build file list for this split (e.g., 02691156/xxxx)
        split_files = self._load_split_file_list(self.split)
        # Keep only selected categories
        split_files = [p for p in split_files if p.split('/')[0] in self.filter_synsets]

        # Build absolute paths and optional .seg paths (if present)
        self.data_paths: List[str] = []
        self.seg_paths: List[Optional[str]] = []
        for rel in split_files:
            pts = os.path.join(self.root, rel + '.txt')
            seg = os.path.join(self.root, rel.replace('/points', '/point_labels') + '.seg')  # in case your list uses that style
            self.data_paths.append(pts)
            self.seg_paths.append(seg if os.path.exists(seg) else None)

        if len(self.data_paths) == 0:
            raise RuntimeError(
                f"No files found for split='{self.split}' under root='{self.root}'. "
                f"Check that train_test_split JSONs match on-disk paths."
            )

        # Category (synset) -> index mapping
        self.cat_to_idx = {syn: i for i, syn in enumerate(sorted(self.filter_synsets))}
        # Precompute per-item category index from its path
        self.category_labels = [self.cat_to_idx[os.path.normpath(p).split(os.sep)[-2]] for p in self.data_paths]

        # ---- Global label discovery & remap (done once, across train+test) ----
        if ShapeNetPart._GLOBAL_REMAP is None:
            # Build over train+test so num_parts is stable
            all_files = self._load_split_file_list('train') + self._load_split_file_list('test')
            # unique labels across all selected categories (or across all if no filter)
            all_files = [p for p in all_files if p.split('/')[0] in self.filter_synsets or category is None]

            global_label_set = set()
            for rel in all_files:
                pts_path = os.path.join(self.root, rel + '.txt')
                seg_path = os.path.join(self.root, rel.replace('/points', '/point_labels') + '.seg')
                if os.path.exists(seg_path):
                    # Read integer labels from .seg
                    try:
                        labs = np.loadtxt(seg_path, dtype=int)
                        if labs.ndim == 0:
                            labs = np.array([int(labs)])
                        global_label_set.update(np.unique(labs).astype(int).tolist())
                        continue
                    except Exception:
                        pass  # fall back to .txt last column

                # Fallback: read last column (part_id) from .txt (7th column)
                try:
                    # Only last column for speed
                    labs = np.loadtxt(pts_path, dtype=float, usecols=(-1,))
                    if labs.ndim == 0:
                        labs = np.array([labs])
                    # Convert robustly to ints
                    labs = np.rint(labs).astype(int)
                    global_label_set.update(np.unique(labs).tolist())
                except Exception:
                    # If we cannot read, skip this file
                    continue

            if len(global_label_set) == 0:
                raise RuntimeError("Could not discover any segmentation labels. "
                                   "Are your .txt files 7-column (xyz nx ny nz label), "
                                   "or do you have .seg files?")

            sorted_labels = sorted(global_label_set)
            remap = {orig: i for i, orig in enumerate(sorted_labels)}

            ShapeNetPart._GLOBAL_LABEL_SET = global_label_set
            ShapeNetPart._GLOBAL_REMAP = remap
            ShapeNetPart._GLOBAL_NUM_PARTS = len(sorted_labels)

        # Public attributes expected by your loader
        self.num_parts: int = int(ShapeNetPart._GLOBAL_NUM_PARTS)
        self.label_names: Dict[int, str] = {new_id: f"part_{orig_id}"
                                            for orig_id, new_id in ShapeNetPart._GLOBAL_REMAP.items()}

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        pts_path = self.data_paths[idx]
        seg_path = self.seg_paths[idx]
        cat_idx = self.category_labels[idx]

        # Load points (+ normals) and labels
        points, labels = self._load_points_and_labels(pts_path, seg_path)

        # Sample points
        n = points.shape[0]
        replace = n < self.num_points
        choice = np.random.choice(n, self.num_points, replace=replace)
        points = points[choice]
        labels = labels[choice]

        # Normalize to unit sphere
        points = self._normalize_points(points)

        # Features: [C, N]
        if self.normal and points.shape[1] >= 6:
            feats = torch.from_numpy(points[:, :6]).float().transpose(1, 0)
        else:
            feats = torch.from_numpy(points[:, :3]).float().transpose(1, 0)

        # Remap labels to contiguous global IDs
        remap = ShapeNetPart._GLOBAL_REMAP
        labels = np.vectorize(remap.get, otypes=[np.int64])(labels)
        labels = torch.from_numpy(labels).long()

        cat_idx = torch.tensor(cat_idx, dtype=torch.long)
        return feats, labels, cat_idx

    def _load_categories(self) -> Tuple[Dict[str, str], List[str]]:
        """
        Returns:
            name2synset: {'Airplane':'02691156', ...}
            synset_list: ['02691156','02773838',...]
        """
        mapping = {}
        fp = os.path.join(self.root, 'synsetoffset2category.txt')
        if not os.path.exists(fp):
            raise FileNotFoundError(f"synsetoffset2category.txt not found under: {self.root}")

        with open(fp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                name, synset = parts
                mapping[name] = synset
        synsets = sorted(mapping.values())
        return mapping, synsets
    def _load_points_and_labels(self, pts_path: str, seg_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single shape's points and per-point labels.

        pts_path: .../<synset>/<id>.txt  with columns: x y z nx ny nz [label]
        seg_path: optional .../<synset>/point_labels/<id>.seg (one int label per point)

        Returns:
            points  (N,6) if normals available else (N,3)
            labels  (N,)  int, NOT yet remapped to [0..num_parts-1]
        """
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"Missing points file: {pts_path}")

        # Load the .txt as float
        try:
            arr = np.loadtxt(pts_path).astype(float)
            if arr.ndim == 1:
                arr = arr[None, :]  # ensure 2D for single-line files
        except Exception as e:
            raise RuntimeError(f"Failed to read points from {pts_path}: {e}")

        labels: Optional[np.ndarray] = None

        # Prefer .seg if provided and valid
        if seg_path is not None and os.path.exists(seg_path):
            try:
                labels = np.loadtxt(seg_path, dtype=int)
                if labels.ndim == 0:  # single value edge case
                    labels = np.array([int(labels)])
            except Exception:
                labels = None  # fall back to embedded last column

        # If no .seg, expect last column of .txt to be the label
        if labels is None:
            if arr.shape[1] >= 7:
                labels = np.rint(arr[:, -1]).astype(int)  # cast robustly like 11.000000 -> 11
                arr = arr[:, :-1]  # drop label column from features
            else:
                raise RuntimeError(
                    f"No .seg for {pts_path} and file has only {arr.shape[1]} columns; "
                    f"expected 7 (xyz nx ny nz label)."
                )

        # Keep xyz(+normals)
        if arr.shape[1] >= 6:
            points = arr[:, :6]  # xyz + normals
        else:
            points = arr[:, :3]  # xyz only

        if points.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"Points/labels length mismatch for {pts_path}: "
                f"{points.shape[0]} pts vs {labels.shape[0]} labels."
            )

        return points, labels

    def _load_split_file_list(self, split: str) -> List[str]:
        """
        Returns a list of normalized entries like '02958343/abcdef...' (no extension).
        Accepts many JSON variants:
        - '02958343/abcdef'
        - '02958343/points/abcdef'
        - '02958343/abcdef.txt'
        - '02958343/point_labels/abcdef.seg'
        - 'shapenetcore_partanno_segmentation_benchmark_v0_normal/02958343/abcdef'
        """
        split_dir = os.path.join(self.root, 'train_test_split')
        fname = f'shuffled_{split}_file_list.json'
        fp = os.path.join(split_dir, fname)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing split file list: {fp}")

        with open(fp, 'r') as f:
            raw_entries = json.load(f)

        def normalize_entry(e: str) -> str:
            e = str(e).strip().replace('\\', '/').strip('/')

            if e.startswith('shape_data/'):
                e = e[len('shape_data/'):]

            # Also handle other known prefixes (keep these)
            ds_prefix = 'shapenetcore_partanno_segmentation_benchmark_v0_normal/'
            if e.startswith(ds_prefix):
                e = e[len(ds_prefix):]

            # Remove mode subfolders if present (robustness)
            e = e.replace('/points/', '/').replace('/point_labels/', '/')
            e = e.replace('/points', '').replace('/point_labels', '')

            # Strip file extensions if any
            if e.endswith('.txt'):
                e = e[:-4]
            if e.endswith('.seg'):
                e = e[:-4]
            return e


        entries = [normalize_entry(e) for e in raw_entries]

        # Keep only entries that actually exist on disk after normalization
        out = []
        for rel in entries:
            pts = os.path.join(self.root, rel + '.txt')
            if os.path.exists(pts):
                out.append(rel)

        if not out:
            candidates = []
            for syn in sorted(set(p.split('/')[0] for p in entries if '/' in p)):
                syn_dir = os.path.join(self.root, syn)
                if os.path.isdir(syn_dir):
                    for fn in os.listdir(syn_dir):
                        if fn.endswith('.txt'):
                            candidates.append(f"{syn}/{os.path.splitext(fn)[0]}")
            # Intersect by basename presence in original entries (after normalization)
            names_in_json = set(p.split('/')[-1] for p in entries if '/' in p)
            out = [p for p in candidates if p.split('/')[-1] in names_in_json]

        return out


    @staticmethod
    def _normalize_points(points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0, keepdims=True)
        xyz = xyz - centroid
        scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if scale > 0:
            xyz = xyz / scale
        out = points.copy()
        out[:, :3] = xyz
        return out
