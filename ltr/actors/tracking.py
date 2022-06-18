from . import BaseActor
import torch
import numpy as np
import ltr.data.processing_utils as prutils


class TranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, settings=None):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats


class BaselineTranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, settings=None):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'], None, None, None, None)

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            _, h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i, -1]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        if 0:
            from matplotlib import pyplot as plt
            import os
            plt.subplot(121);plt.imshow(data['search_images'][0,0].permute(1,2,0).cpu());plt.subplot(122);plt.imshow(data['bump'][0,0].squeeze().cpu());plt.show()
            im = 0
            # for idx in range(3*8):
            #     if idx % 8 == 0:
            #         im+=1
            #     mi = idx % 8
            #     plt.subplot(3,8,idx + 1)
            #     plt.axis("off")

            #     plt.imshow(data['search_images'][im, mi].permute(1,2,0).cpu())
            for idx in range(15):
                plt.subplot(4,4,idx + 1)
                plt.axis("off")
                plt.imshow(data['search_images'][im, idx].permute(1,2,0).cpu())

            plt.subplot(4,4,16)
            plt.axis("off")
            plt.imshow(data['template_images'][im].permute(1,2,0).cpu())

            plt.show()
            os._exit(1)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets, None, None)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats


class CircuitTranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, settings=None):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time

        # generate labels
        targets, boxes = [], []
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            _, h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i, -1]
            target = {}
            target_origin = target_origin.reshape([1,-1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # if 1:
        #     from matplotlib import pyplot as plt
        #     import os
        #     im = 0
        #     for idx in range(8):
        #         plt.subplot(2,8,idx + 1)
        #         plt.axis("off")
        #         plt.imshow(data['search_images'][im, idx].permute(1,2,0).cpu())
        #         plt.subplot(2,8,idx + 1 + 8)
        #         plt.axis("off")
        #         plt.imshow(data['bump'][im, idx].squeeze().cpu())


        #    plt.show()
        #      os._exit(1)

        boxes = torch.stack(boxes, 0)
        bumps = data['bump'][:, 0]
        outputs = self.net(data['search_images'], data['template_images'], samp_idx=None, labels=bumps, settings=settings, boxes=boxes)  # boxes=boxes)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        # loss_dict = self.objective(outputs, targets, boxes, data['search_visible'])
        loss_dict = self.objective(outputs, targets, data['bump'], data['search_visible'])
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'Loss/circuit_iou': loss_dict['loss_iou_circuit'].item(),
                 'loss/circuit_giou': loss_dict['loss_giou_circuit'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats


class OldCircuitTranstActor(BaseActor):
    """ Actor for training the TransT"""
    def __call__(self, data, settings):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        sequence_length = data['search_images'].shape[1]
        num_sequences = data['search_images'].shape[0]

        # generate labels
        samp_idx = None  # np.random.randint(low=settings.search_gap, high=sequence_length)
        targets, boxes = [], []
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            # Sequences are expected
            _, h, w =data['search_images'][i][0].shape
            target_origin = targets_origin[i]

            target = {}
            # target_origin = target_origin.reshape([1,-1])
            target_origin[:, 0] += target_origin[:, 2] / 2
            target_origin[:, 0] /= w
            target_origin[:, 1] += target_origin[:, 3] / 2
            target_origin[:, 1] /= h
            target_origin[:, 2] /= w
            target_origin[:, 3] /= h
            target['boxes'] = target_origin[-1][None]
            boxes.append(target_origin)  # Whole sequence for the RNN
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)
        boxes = torch.stack(boxes, 0)

        # Sample the search_images inds for the transformer
        # bumps = data['bump'][:, 0]
        if 0:
            from matplotlib import pyplot as plt
            import os
            plt.subplot(121);plt.imshow(data['search_images'][0,0].permute(1,2,0).cpu());plt.subplot(122);plt.imshow(data['bump'][0,0].squeeze().cpu());plt.show()
            im = 0
            # for idx in range(3*8):
            #     if idx % 8 == 0:
            #         im+=1
            #     mi = idx % 8
            #     plt.subplot(3,8,idx + 1)
            #     plt.axis("off")

            #     plt.imshow(data['search_images'][im, mi].permute(1,2,0).cpu())
            for idx in range(10):
                plt.subplot(2,5,idx + 1)
                plt.axis("off")
                plt.imshow(data['search_images'][im, idx].permute(1,2,0).cpu())

            plt.subplot(4,4,16)
            plt.axis("off")
            plt.imshow(data['template_images'][im].permute(1,2,0).cpu())

            plt.show()
            os._exit(1)
        outputs = self.net(data['search_images'], data['template_images'], samp_idx=samp_idx, labels=data['bump'][:, 0], settings=settings, boxes=boxes)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        # loss_dict = self.objective(outputs, targets, boxes, data['search_visible'])
        loss_dict = self.objective(outputs, targets, boxes, data['search_visible'])
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'Loss/circuit_iou': loss_dict['loss_iou_circuit'].item(),
                 'Loss/circuit_giou': loss_dict['loss_giou_circuit'].item(),
                 # 'Loss/predicted_error': loss_dict['predicted_error'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats
