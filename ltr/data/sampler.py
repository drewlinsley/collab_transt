import os
import random
import torch.utils.data
from pytracking import TensorDict
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def no_processing(data):
    return data

class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of a set of template frames and search frames, used to train the TransT model.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'template frames' and
    'search frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the search frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _sample_visible_ids_ar(self, visible, num_ids=1, min_id=None, max_id=None, num_search=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if len(visible) > (i + num_search) and (i - num_search) > 0 and ((visible[i] and visible[i + num_search]) or (visible[i] and visible[i - num_search]))]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _sample_seq_ids(self, visible, num_ids=1, min_id=None, max_id=None, random_speeds=True):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if max_id is None:
            max_id = len(visible)
        if max_id > len(visible):
            max_id = len(visible)
        if max_id < 0:
            max_id = 0

        if min_id > len(visible):
            min_id = len(visible)
        if min_id is None:  #  or min_id < 0:
            min_id = 0
        if min_id < 0:
            min_id = 0

        sign = 1
        if min_id > max_id:
            sign = -1
            valid_ids = visible[max_id:min_id]
        else:
            valid_ids = visible[min_id:max_id]
        if len(valid_ids) == 0:
            return None

        if random_speeds:
            # Choose random offset for sequence
            max_offset = (max(max_id, min_id) - min(max_id, min_id)) // num_ids
            if max_offset == 0:
                return None  # Sequence is too short
            max_offset = min(max_offset, 3)  # Allow speed=1 or speed=2 or speed=3
            weights = np.ones(max_offset)
            offset = np.random.choice(np.arange(1, max_offset + 1, dtype=int), p=weights / weights.sum()) * sign
            idx = np.arange(min_id, (min_id + (num_ids * offset)), offset)  # moving up/down based on offset sign
        else:
            idx = np.arange(min_id, min_id + sign * num_ids, sign)

        if not visible[idx[-1]]:
            return None  # Need a label on final frame. This is passed to the transformer.
        return idx

    def sample_seq(self, dataset, vis_thresh, is_video_dataset):
        enough_visible_frames = False
        while not enough_visible_frames:
            if 0:  # self.frame_sample_mode == 'rnn_causal':
                # Sample the harder half of the sequences
                seq_id = random.choices(cid, np.ones_like(cid))[0]
            else:
                # Sample a sequence
                seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= vis_thresh

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, seq_info_dict, visible, enough_visible_frames

    def get_data_difficulty(self, dataset):
        zs, bs = [], []
        for idx in range(dataset.get_num_sequences()):
            seq_info_dict = dataset.get_sequence_info(idx)
            visible = seq_info_dict['visible'].float()
            bboxs = seq_info_dict['bbox'].float()
            z = visible.mean()  #  / visible.std()
            b = (np.abs(np.diff(bboxs[:, 0])) + np.abs(np.diff(bboxs[:, 1]))).mean()
            zs.append(z)
            bs.append(b)
        zs = torch.stack(zs).numpy()
        bs = np.asarray(bs)
        challenges = bs  # challenges = bs * (1 - sigmoid(zs))  # Velocity weighted by occlusion -> higher = faster + more occluded
        challenges = np.argsort(challenges)[::-1]
        np.save("data_stats/{}".format(dataset.name), challenges)
        return challenges

    def __getitem__(self, index, vis_thresh=20, challenge_thresh=0.5):  # Dropped vis_thresh from 20 -> 10
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Order dataset by difficulty
        # ids = []
        # for dataset in self.datasets:
        #     if os.path.exists("data_stats/{}.npy".format(dataset.name)):
        #         ids.append(np.load("data_stats/{}.npy".format(dataset.name)))
        #     else:
        #         ids.append(self.get_data_difficulty(dataset))

        # Select a dataset
        idx = random.choices(range(len(self.p_datasets)), self.p_datasets)[0]
        dataset = self.datasets[idx]
        # cid = ids[idx][:int(challenge_thresh * len(ids[idx]))]
        # dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        seq_id, seq_info_dict, visible, enough_visible_frames = self.sample_seq(dataset, vis_thresh, is_video_dataset)
        count = 0
        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'interval_sorted':
                # Sample frame numbers within interval defined by the first frame
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_template_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + gap_increase)
                    if extra_template_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                                                              min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase)
                    if template_frame_ids[0] > max(search_frame_ids):
                        search_frame_ids = sorted(search_frame_ids)
                    else:
                        search_frame_ids = sorted(search_frame_ids)[::-1]  # Sort for the RNN
                    gap_increase += 5  # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
            elif self.frame_sample_mode == 'rnn_causal':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                # visible = torch.ones_like(visible)  # Force everything visible
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                             max_id=len(visible) - self.num_search_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    template_frame_ids = base_frame_id + prev_frame_ids
                    # search_frame_ids = np.arange(template_frame_ids[0] + 1, template_frame_ids[0] + 1 + self.num_search_frames)  # Rather than sample just take the rest of the sequence in order.
                    # Sample from template to the next self.num_search_frames
                    search_frame_ids = self._sample_seq_ids(visible, min_id=template_frame_ids[0] + 1,  # template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_search_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5


            elif self.frame_sample_mode == 'rnn_interval':
                # Sample search and template frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                # visible = torch.ones_like(visible)  # Force everything visible
                gap_increase = 0
                count = 0
                num_search_frames = self.num_search_frames
                import pdb;pdb.set_trace()
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids_ar(visible, num_ids=1, num_search=num_search_frames)
                    extra_template_frame_ids = self._sample_visible_ids_ar(visible, num_ids=self.num_template_frames - 1, num_search=num_search_frames,
                                                                     min_id=base_frame_id[
                                                                                0] - self.max_gap - 1 - gap_increase,
                                                                     max_id=base_frame_id[
                                                                                0] + self.max_gap + 1 + gap_increase)
                    # if extra_template_frame_ids is None:
                    #     gap_increase += 5
                    #     continue
                    template_frame_ids = base_frame_id + extra_template_frame_ids
                    # 4 cases: min -> mid, mid -> max, max -> mid, mid -> min
                    min_to_mid = 1 if (template_frame_ids[0] - self.max_gap - gap_increase) > 0 else 0
                    mid_to_max = 1 if (template_frame_ids[0] + self.max_gap + gap_increase) < len(visible) else 0
                    max_to_mid = 1 if (template_frame_ids[0] + self.max_gap + gap_increase) < len(visible) else 0
                    mid_to_min = 1 if (template_frame_ids[0] - self.max_gap - gap_increase) > 0 else 0
                    # case = random.choices(np.arange(5), [min_to_mid, mid_to_max, max_to_mid, mid_to_min, 1])[0]
                    if sum([0, mid_to_max, 0, mid_to_min]) == 0:
                        # gap_increase += 5  # Increase gap until a frame is found
                        # count += 1
                        # if count > 4:
                            # num_search_frames = num_search_frames // 2
                            # Let's just sample a new dataset.
                        seq_id, seq_info_dict, visible, enough_visible_frames = self.sample_seq(dataset, vis_thresh, is_video_dataset)
                        continue

                    case = random.choices(np.arange(4), [0, mid_to_max, 0, mid_to_min])[0]
                    if case == 0:
                        search_frame_ids = self._sample_seq_ids(visible, min_id=template_frame_ids[0] - self.max_gap - gap_increase,  # template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0],
                                                              num_ids=num_search_frames)
                    elif case == 1:
                        search_frame_ids = self._sample_seq_ids(visible, min_id=template_frame_ids[0],  # template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=num_search_frames)
                    elif case == 2:
                        search_frame_ids = self._sample_seq_ids(visible, min_id=template_frame_ids[0] + self.max_gap + gap_increase,  # template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0],
                                                              num_ids=num_search_frames)
                    elif case == 3:
                        search_frame_ids = self._sample_seq_ids(visible, min_id=template_frame_ids[0],  # template_frame_ids[0] + 1,
                                                              max_id=template_frame_ids[0] - self.max_gap - gap_increase,
                                                              num_ids=num_search_frames)

                # if search_frame_ids is None or np.any(search_frame_ids < 0):
                #     import pdb;pdb.set_trace()
            else:
                raise NotImplementedError(self.frame_sample_mode)

        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            template_frame_ids = [1] * self.num_template_frames
            search_frame_ids = [1] * self.num_search_frames

        template_frames, template_anno, meta_obj_template = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
        search_frames, search_anno, meta_obj_search = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
        data = TensorDict({'template_images': template_frames,
                           'template_anno': template_anno['bbox'],
                           'search_images': search_frames,
                           'search_visible': visible[search_frame_ids],
                           'search_anno': search_anno['bbox']})
        return self.processing(data)



class TransTSampler(TrackingSampler):
    """ See TrackingSampler."""

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames=1, num_template_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_search_frames=num_search_frames, num_template_frames=num_template_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)