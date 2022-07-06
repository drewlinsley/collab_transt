import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings
from glob import glob


class Dream(BaseVideoDataset):
    """ Dream data
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the dream dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().dream_dir if root is None else root
        super().__init__('Dream', root, image_loader)

        # Keep a list of all classes
        self.class_list = ["neuron"]  # [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list, self.annos, self.file_paths = self._build_sequence_list(vid_ids, split, root)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = {self.class_list[0]: self.sequence_list}  # self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None, root=None):
        if split is None or split == "train":
            # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            import pdb;pdb.set_trace()
            files = glob(os.path.join(root, "*"))
            files = [x for x in files if "." not in x]
            annotations = pandas.read_csv(os.path.join(root, "formatted_data.csv"))

            # Filter by well
            file_wells = [x.split(os.path.sep)[-1] for x in files]
            annos_files = {}
            file_paths = {}
            for well, file in zip(file_wells, files):
                data = annotations[annotations.well == well]

                if len(data):
                    # Then sort by time
                    data = data.sort_values("time")

                    # Now package into a list of lists, with each list corresponding to a different tracked cell
                    objects = data.object.unique()
                    tracks = []
                    for obj in objects:
                        coords = data[data.object == obj][["w", "h", "width", "height"]].values.tolist()
                        tracks.append(coords)

                    # Store in a dict
                    annos_files[well] = tracks
                    # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
                    file_paths[well] = file
                else:
                    # Remove this well from the dict
                    pass

        else:
            raise ValueError('Set either split_name or vid_ids.')
        sequence_list = {idx: k for idx, k in enumerate(annos_files.keys())}
        return sequence_list, annos_files, file_paths

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'dream'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = "neuron"  # seq_name.split('-')[0]
        vid_id = seq_name  # seq_name.split('-')[1]

        return os.path.join(self.root, "cell_video_{}.npy".format(vid_id)), seq_name
        # return os.path.join(self.root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        seq_path, well_name = self._get_sequence_path(seq_id)
        bboxs = self.annos[well_name]
        import pdb;pdb.set_trace()
        # Grab a random object
        n = len(bboxs)
        if n == 0:
            import pdb;pdb.set_trace()
        bbox = bboxs[np.random.permutation(n)[0]]
        bbox = torch.tensor(bbox)
        # bbox = self._read_bb_anno(seq_path)

        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible = self._read_target_visible(seq_path) & valid.byte()
        visible, valid = torch.ones(len(bbox)), torch.ones(len(bbox))

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{}.jpg'.format(frame_id))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, well_name = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        import pdb;pdb.set_trace()
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        # frame_list = np.load(self.file_paths[well_name]).tolist()  # Change to jpegs next!

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        import pdb;pdb.set_trace()
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
