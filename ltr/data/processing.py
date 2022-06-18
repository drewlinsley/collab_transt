import torch
import numpy as np
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'template':  transform if template_transform is None else template_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class TransTProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor, rand=False,
                 mode='pair', label_function_params=False, occlusion=False, joint=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.label_function_params = label_function_params
        self.joint = joint
        self.rand = rand
        self.occlusion = occlusion

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.label_function_params['output_sz'],
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _get_jittered_box(self, box, mode, rand_size=None, rand_center=None):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """
        if rand_size is None:
            rand_size = torch.randn(2)
        if rand_center is None:
            rand_center = torch.rand(2)
        jittered_size = box[2:4] * torch.exp(rand_size * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (rand_center - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'])
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'], new_roll=False)

        # self.label_function_params = {"kernel_sz": 4, "feature_sz": 256, "output_sz": self.search_sz, "end_pad_if_even": False, "sigma_factor": 0.05} 
        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            if self.rand:
                rand_size_a = torch.randn(2)
                rand_center_a = torch.rand(2)
                rand_size_b = torch.randn(2)
                rand_center_b = torch.rand(2)

                # Linearly interpolate from 0 to rand_size/center
                size_step = torch.tensor(np.linspace(rand_size_a, rand_size_b, len(data[s + '_anno'])))
                center_step = torch.tensor(np.linspace(rand_center_a, rand_center_b, len(data[s + '_anno'])))
                jittered_anno = [self._get_jittered_box(a, s, rand_size=rs, rand_center=rc) for a, rs, rc in zip(data[s + '_anno'], size_step, center_step)]
            else:
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box

            if s == 'search':
                if torch.any(data['search_visible'] == 0):
                    # For empty annos, use the most recent crop box coords.
                    filler_anno = jittered_anno[0]
                    # assert filler_anno.sum(), "First frame was empty."  # Only last frame matters
                    filler_jitter = data[s + '_anno'][0]
                    for mi in range(len(data['search_visible'])):
                        if data['search_visible'][mi] == 0:
                            jittered_anno[mi] = filler_anno
                            data[s + '_anno'][mi] = filler_jitter
                        else:
                            filler_anno = jittered_anno[mi]
                            filler_jitter = data[s + '_anno'][mi]
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.search_sz)
                # except:
                #     print("Jitter")
                #     print(jittered_anno)
                #     print("Regular")
                #     print(data[s + '_anno'])
                #     print("data['search_visible']")
                #     print(data['search_visible'])
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError
            # Boxes is columns,rows,column-offset,row-offset

            # Apply transforms
            if s == "search" and self.occlusion:
                maybe_occlusion = np.random.rand() > 0.5
                crops = list(crops)
                min_size = 1  # 10
                min_frames = 7  # 10  # When should the occlusion start
                if maybe_occlusion:
                    # import pdb;pdb.set_trace()
                    # rand_frames_len = np.random.randint(low=0, high=len(crops) - min_frames)  # len(data[s + '_images']) - min_frames)
                    # rand_frames_start = np.random.randint(low=min_frames, high=len(crops) - rand_frames_len)  # data[s + '_images']) - rand_frames_len)
                    crop_len = len(crops)
                    rand_frames_start = np.random.randint(low=min_frames, high=crop_len)
                    rand_frames_len = crop_len - rand_frames_start
                    top_side = rand_frames_start % 2

                    # Find the box in the first from, and use this to construct occluder
                    start_box = boxes[rand_frames_start].int()
                    crop_shape = crops[0].shape  # data[s + '_images'][0].shape
                    apply_occlusion = False
                    pass_check = start_box[2] // 2 > min_size and start_box[3] // 2 > min_size and crops[0].shape[0] > min_size and crops[0].shape[1] > min_size
                    if top_side and pass_check:
                        # These are row inds
                        rand_start = np.random.randint(low=0, high=start_box[3] - min_size - 1)
                        if rand_start > start_box[3] // 2:
                            margin = np.copy(rand_start)
                            rand_start = np.random.randint(low=0, high=margin - min_size)
                            rand_extent = margin - rand_start
                        else:
                            remainder = np.maximum(start_box[3] - rand_start, min_size)
                            mc, xc = np.minimum(rand_start, remainder), np.maximum(rand_start, remainder)
                            if mc == xc: 
                                xc += 1
                                rand_extent = mc + 1
                            else:
                                rand_extent = np.random.randint(low=mc, high=xc)

                        # rand_mask = (np.random.rand(rand_extent, crop_shape[1], crop_shape[2]) * 128) + 128
                        rand_start += start_box[1]
                        if rand_start + rand_extent < crops[0].shape[0] and rand_start > 0:
                            apply_occlusion = True
                    elif not top_side and pass_check:
                        # These are width inds
                        rand_start = np.random.randint(low=0, high=start_box[2] - min_size - 1)
                        if rand_start > start_box[2] // 2:
                            margin = np.copy(rand_start)
                            rand_start = np.random.randint(low=0, high=margin - min_size)
                            rand_extent = margin - rand_start
                        else:
                            # remainder = np.maximum((start_box[2] - margin - rand_start), min_size + 1)
                            remainder = np.maximum(start_box[3] - rand_start, min_size)
                            mc, xc = np.minimum(rand_start, remainder), np.maximum(rand_start, remainder)
                            if mc == xc:
                                xc += 1
                                rand_extent = mc + 1
                            else:
                                rand_extent = np.random.randint(low=mc, high=xc)

                        # rand_mask = (np.random.rand(crop_shape[0], rand_extent, crop_shape[2]) * 128) + 128
                        rand_start += start_box[0]
                        if rand_start + rand_extent < crops[0].shape[1] and rand_start > 0:
                            apply_occlusion = True
                    if apply_occlusion:
                        # print("applying occlusion")
                        # for bidx in range(rand_frames_start, rand_frames_start + rand_frames_len):
                        for bidx in range(rand_frames_start, crop_len):
                        # Apply an occluder to a random location in a random chunk of the video
                        # data[s + '_images'][bidx] = data[s + '_images'][bidx] + mask
                            if top_side:
                                shuffle_box = crops[bidx][rand_start: rand_start + rand_extent]
                                shuffle_shape = shuffle_box.shape
                                shuffle_box = shuffle_box.reshape(-1, shuffle_shape[-1])  # channels last
                                shuffle_box = shuffle_box[np.random.permutation(shuffle_shape[0] * shuffle_shape[1])]
                                crops[bidx][rand_start: rand_start + rand_extent] = shuffle_box.reshape(shuffle_shape)  #  rand_mask
                            else:
                                shuffle_box = crops[bidx][:, rand_start: rand_start + rand_extent]
                                shuffle_shape = shuffle_box.shape
                                shuffle_box = shuffle_box.reshape(-1, shuffle_shape[-1])  # channels last
                                shuffle_box = shuffle_box[np.random.permutation(shuffle_shape[0] * shuffle_shape[1])]
                                crops[bidx][:, rand_start: rand_start + rand_extent] = shuffle_box.reshape(shuffle_shape)  #  rand_mask
                            # from matplotlib import pyplot as plt
                            # plt.imshow(crops[bidx])
                            # plt.title("frame: {} topside: {} start: {} extent: {}".format(bidx, top_side, rand_start, rand_extent))
                            # plt.show()
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=self.joint)
            if s == "search":
                im_shape = [len(data[s + '_images']), 1] + [x for x in data[s + '_images'][0].shape[1:]]
                bumps = torch.zeros(im_shape, device=data[s + '_images'][0].device).float()
                for bidx in range(bumps.shape[0]):
                    box = boxes[bidx].int()
                    bumps[bidx, :, box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = 1
                data["bump"] = bumps  # self._generate_label_function(torch.cat(boxes, 0) / self.search_sz) 
        self.prev_annos = jittered_anno

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        return data
