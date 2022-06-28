import torch.nn as nn
from ltr import model_constructor

import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2,
                       accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.loss.matcher import build_matcher
# from ltr.models.neck.circuit_featurefusion_network_test1 import build_featurefusion_network
# from ltr.models.neck.circuit_featurefusion_encoder_test_mult import build_featurefusion_network
# from ltr.models.neck.circuit_featurefusion_network_mixed import build_featurefusion_network
# from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.neck.circuit_featurefusion_encoder_q import build_featurefusion_network
from ltr.admin.loading import load_weights
# from ltr.models.neck.circuit_rnn import ConvGRUCell
from ltr.models.neck.circuit_rnn_pt import ConvGRUCell
import ltr.data.processing_utils as prutils
from matplotlib import pyplot as plt


class TransT(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes, input_dim=32, rnn_dims=32, timesteps=8):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        # self.circuit = circuit
        self.rnn_dims = rnn_dims
        hidden_dim = featurefusion_network.d_model
        self.hidden_dim = hidden_dim
        # self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.exc_bbox = MLP(hidden_dim, hidden_dim, 4, 3)
        self.circuit_exc_bbox_1 = nn.Conv3d(rnn_dims, 1, kernel_size=[1, 1, 1], padding=[1//2, 1//2, 1//2])
        # self.circuit_exc_bbox_2 = nn.Conv3d(16, 1, kernel_size=[1, 3, 3], padding=[1//2, 3//2, 3//2])
        self.height = 32
        self.circuit_inp_f = 1024
        self.circuit_exc_bbox_3 = nn.Linear(self.height * self.height, 4)

        # self.rnn_embed = MLP(hidden_dim, hidden_dim * 2, hidden_dim, 3)
        self.class_embed = MLP(hidden_dim * 1, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim * 1, hidden_dim, 4, 3)
        # self.rnn_embed = MLP(self.rnn_dims, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.circuit_rnn_decode_1 = nn.Conv2d(self.rnn_dims, hidden_dim, kernel_size=1, padding=1//2)
        self.circuit_rnn_decode_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=1//2)
        self.circuit_rnn_decode_3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=1//2)
        self.nl = F.relu  # F.softplus
        self.cnl = F.softplus
        self.backbone = backbone
        self.reset_hidden = False
        self.fr = False
        self.vj_pen = False
        self.timesteps = timesteps
        self.circuit_proj = nn.Conv3d(self.circuit_inp_f, self.rnn_dims, kernel_size=1)
        self.circuit_td_1 = ConvGRUCell(input_dim=self.rnn_dims, hidden_dim=self.rnn_dims, kernel_size=1, padding_mode='zeros', batchnorm=True, use_attention=True, timesteps=self.timesteps)
        self.circuit_1 = ConvGRUCell(input_dim=self.rnn_dims, hidden_dim=self.rnn_dims, kernel_size=5, padding_mode='zeros', batchnorm=True, use_attention=True, timesteps=self.timesteps)
        self.circuit_exc_1_init = nn.Conv2d(1, self.rnn_dims, kernel_size=1)
        self.circuit_inh_1_init = nn.Conv2d(1, self.rnn_dims, kernel_size=1)
        self.circuit_td_inh_1_init = nn.Conv2d(self.rnn_dims, self.rnn_dims, kernel_size=1)
        self.circuit_gate = nn.Conv2d(self.rnn_dims ** 2 + 1, hidden_dim, kernel_size=3, padding=3 // 2)
        # self.circuit_gate_readout = nn.Conv2d(1024, 1, kernel_size=1, padding=1 // 2)
        # self.circuit_gate_readout = nn.Linear(1, 1)

        self.circuit_td_inh_proj = nn.Conv2d(self.rnn_dims, self.rnn_dims, kernel_size=1)
        self.circuit_tf_codeswitch = nn.Conv2d(hidden_dim, self.rnn_dims, kernel_size=3, padding=3 // 2)
        self.count = 0

    def _generate_label_function(self, target_bb, sigma, kernel, feature, output_sz, end_pad_if_even, target_absent=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), sigma,
                                                      kernel,
                                                      feature, output_sz,
                                                      end_pad_if_even=end_pad_if_even)
        if target_absent is not None:
            gauss_label *= (1 - target_absent).view(-1, 1, 1).float()
        return gauss_label

    def forward(self, search, template, samp_idx, labels, settings, boxes):
        """ The forward expects a NestedTensor, which consists of:
               - search.tensors: batched images, of shape [batch_size x 3 x H_search x W_search]
               - search.mask: a binary mask of shape [batch_size x H_search x W_search], containing 1 on padded pixels
               - template.tensors: batched images, of shape [batch_size x 3 x H_template x W_template]
               - template.mask: a binary mask of shape [batch_size x H_template x W_template], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits for all feature vectors.
                                Shape= [batch_size x num_vectors x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all feature vectors, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image.

        """
        # Reshape search into a 4D tensor
        search_shape = [int(x) for x in search.shape]
        searcht = search.view([search_shape[0] * search_shape[1]] + search_shape[2:])
        if not isinstance(searcht, NestedTensor):
            searcht = nested_tensor_from_tensor(searcht)
        if not isinstance(template, NestedTensor):
            templatet = nested_tensor_from_tensor(template)
        with torch.no_grad():
            feature_search, pos_search = self.backbone(searcht)
            feature_template, pos_template = self.backbone(templatet)
        src_search, mask_search = feature_search[-1].decompose()

        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        circuit_input, _ = feature_search[-1].decompose()

        post_src_search_shape = [int(x) for x in src_search.shape]
        post_mask_search_shape = [int(x) for x in mask_search.shape]
        src_search = src_search.view(search_shape[:2] + post_src_search_shape[1:])
        src_template = self.input_proj(src_template)
        mask_search = mask_search.view(search_shape[:2] + post_mask_search_shape[1:])

        circuit_input = circuit_input.view(search_shape[0], search_shape[1], -1, circuit_input.shape[2], circuit_input.shape[3]).permute(0, 2, 1, 3, 4)
        proc_labels = F.interpolate(labels, circuit_input.shape[-2:])
        proc_label_shape = [int(x) for x in proc_labels.shape]
        proc_labels = proc_labels.view([search_shape[0]] + proc_label_shape[1:]).to(src_search.device)  # .mean(2)
        inh_1 = self.cnl(self.circuit_inh_1_init(1 - proc_labels))
        exc_1 = self.cnl(self.circuit_exc_1_init(proc_labels))

        nl_src_search = self.cnl(self.circuit_proj(circuit_input))
        pos_search_shape = [int(x) for x in pos_search[-1].shape]
        pos_search = pos_search[-1].view(search_shape[:2] + pos_search_shape[1:])
        mask_search = mask_search.view(search_shape[:2] + post_mask_search_shape[1:])
        excs, rnn_gates = [], []
        for t in range(nl_src_search.shape[2]):

            # The RNN is going to interact with the final CFA Q.
            # This will be Q + gate * RNN_act. Gate is computed based on the agreement
            # between the RNN and the Q. Once the RNN becomes good enough, the hope is that when agreement
            # is low and Q entropy is high and RNN entropy is low then the gate will also be high.
            # This can be simplified: 

            # Step 1, saliency
            pre_exc_1, inh_1 = self.circuit_1(nl_src_search[:, :, t], excitation=exc_1, inhibition=inh_1, activ=self.cnl)

            # Split off a pair of TransT features and then use the circuit to gate the Qs in the decoder.
            dec_rnn = self.circuit_rnn_decode_1(pre_exc_1)
            prev_hs = self.circuit_rnn_decode_2(self.cnl(dec_rnn))  # * torch.sigmoid(self.rnn_gate(dec_rnn))
            prev_hs = self.circuit_rnn_decode_3(self.cnl(prev_hs))  # * torch.sigmoid(self.rnn_gate(dec_rnn))

            # Pass activities through transformer
            # prev_hs = 1 + (prev_hs.tanh())  # in [0, 2]
            if t > 0:
                # cost_vol = torch.exp(-(res_hs - pre_exc_1) ** 2)  # Correspondence between FB and the predicted modulation
                # This cost vol is all-to-all spatial maps. Then Reshaped so that all features are aggregated.
                # This tells us where we should adjust the Q
                cost_vol = torch.einsum('bik,bjk->bijk', res_hs.view(pre_exc_1.shape[0], pre_exc_1.shape[1], -1), pre_exc_1.view(pre_exc_1.shape[0], pre_exc_1.shape[1], -1)).view(pre_exc_1.shape[0], pre_exc_1.shape[1] ** 2, pre_exc_1.shape[2], pre_exc_1.shape[3])
                # Add the entropy in res_hs as an extra channel, in case it is needed for better gating
                entropy = -(res_hs * (res_hs + 1e-4).log()).sum(dim=1, keepdims=True)
                cost_vol = torch.cat([cost_vol, entropy], 1)
                rnn_gate = self.circuit_gate(cost_vol).sigmoid()
                prev_hs = prev_hs * rnn_gate
            else:
                prev_hs = prev_hs * 0  # Dont allow the RNN on the first frame
            src_input = self.input_proj(src_search[:, t])
            hs, qs = self.featurefusion_network(src_template, mask_template, src_input, mask_search[:, t], pos_template[-1], pos_search[:, t], exc=prev_hs)

            # Step 3, TD-FB incorporating hs to make better modulations of q in the future
            pre_res_hs = qs.squeeze().view(search_shape[0], self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2)
            # pre_res_hs = qs.squeeze().view(search_shape[0], self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2)  # .contiguous()

            # TD from Trans to Circuit
            res_hs = self.cnl(self.circuit_tf_codeswitch(pre_res_hs))
            ##### This pooling is for technical reasons. Ideally dont have to do this but whatever...
            if t == 0:
                td_inh_1 = self.cnl(self.circuit_td_inh_1_init(res_hs))
            exc_1, td_inh_1 = self.circuit_td_1(pre_exc_1, excitation=res_hs, inhibition=td_inh_1, activ=self.cnl)


            if t > 0:
                from matplotlib import pyplot as plt
                plt.subplot(151);plt.imshow(search[0, t].squeeze().permute(1, 2, 0).cpu());
                plt.subplot(152);plt.imshow((src_search[0, t].squeeze()).mean(0).detach().cpu());
                plt.subplot(153);plt.title("Circuit-Transformer agreement", fontsize=6);plt.imshow((rnn_gate[0]).squeeze().detach().cpu().mean(0))
                plt.subplot(154);plt.imshow(-(((prev_hs[0])).squeeze().mean(0).detach().cpu()));plt.title("Circuit Modulation", fontsize=6);
                plt.subplot(155);plt.title("Transformer", fontsize=6);plt.imshow((-hs[0, 0].squeeze().view(1, self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2)).squeeze().mean(0).detach().cpu());  # plt.show()
                plt.show()

            # if t > src_search.shape[1] - 5:
            #     plt.subplot(141);plt.title("Resnet");plt.imshow(search[0, t].squeeze().permute(1, 2, 0).cpu());plt.title("L1");plt.subplot(142);plt.imshow((pre_exc_1[0] ** 2).squeeze().mean(0).detach().cpu());plt.subplot(143);plt.title("L2");plt.imshow((exc_2[0] ** 2).squeeze().mean(0).detach().cpu());plt.subplot(144);plt.title("L1 after TD");plt.imshow((exc_1[0] ** 2).squeeze().mean(0).detach().cpu());
            #     plt.show()
            excs.append(exc_1)

        # Concat exc to hs too
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        excs = torch.stack(excs, 2)
        proc_rnn = self.circuit_exc_bbox_3(self.cnl(self.circuit_exc_bbox_1(excs)).view(excs.shape[0], excs.shape[2], -1)).sigmoid()

        # rnn_gate = self.circuit_gate_readout(self.cnl(cost_vol)).mean((1, 2, 3))  # [:, None]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'hgru_boxes': proc_rnn}  # , 'rnn_gate': rnn_gate}
        return out

    def force_reset(self, label):
        """Quantize self.exc and return it as a label."""
        raise NotImplementedError
        exc = (self.exc ** 2).mean(1).squeeze()
        exc = (exc / exc.max()) > 0.5
        exc = exc.float()
        exc = exc[None, None]
        exc = F.interpolate(exc, label.shape[2:])
        self.exc = exc
        self.inh = exc
        # print("Forced reset")
        self.fr = True

    def reset_states(self):
        self.reset_hidden = True  # exc = None
        self.exc = None
        self.inh = None
        # print("Reset hidden states.")

    def track(self, search, bumps, info):
        imshape = search.shape
        ims = search.clone()  # COMMENT ME
        if not isinstance(search, NestedTensor):
            search = nested_tensor_from_tensor_2(search)
        features_search, pos_search = self.backbone(search)
        feature_template = self.zf
        pos_template = self.pos_template
        src_search, mask_search= features_search[-1].decompose()
        assert mask_search is not None
        src_template, mask_template = feature_template[-1].decompose()
        assert mask_template is not None
        src_template = self.input_proj(src_template)

        circuit_input = src_search
        circuit_input = circuit_input[:, :, None]
        proc_labels = F.interpolate(bumps, circuit_input.shape[-2:])
        proc_label_shape = [int(x) for x in proc_labels.shape]
        proc_labels = proc_labels.view([1] + proc_label_shape[1:]).to(src_search.device)  # .mean(2)

        # Use the circuit to track through the features
        # frame = self.nl(self.circuit_ff_bn(src_search))
        nl_src_search = self.cnl(self.circuit_proj(circuit_input))
        nl_src_search = nl_src_search.squeeze(2)
        if self.reset_hidden:  # not hasattr(self, "exc"):
            # import pdb;pdb.set_trace()
            print("Resetting hidden states")
            inh_1 = self.cnl(self.circuit_inh_1_init(1 - proc_labels))
            exc_1 = self.cnl(self.circuit_exc_1_init(proc_labels))
            td_inh_1 = None  # self.nl(torch.zeros_like(exc_2))
        else:
            exc_1, inh_1 = self.exc_1, self.inh_1
            td_inh_1 = self.td_inh_1
            res_hs = self.res_hs

        pre_exc_1, inh_1 = self.circuit_1(nl_src_search, excitation=exc_1, inhibition=inh_1, activ=self.cnl)
        dec_rnn = self.circuit_rnn_decode_1(pre_exc_1)
        prev_hs = self.circuit_rnn_decode_2(self.cnl(dec_rnn))  # * torch.sigmoid(self.rnn_gate(dec_rnn))
        prev_hs = self.circuit_rnn_decode_3(self.cnl(prev_hs))  # * torch.sigmoid(self.rnn_gate(dec_rnn))

        # Pass activities through transformer
        if td_inh_1 is not None:
            entropy = -(res_hs * (res_hs + 1e-4).log()).sum(dim=1, keepdims=True)
            cost_vol = torch.einsum('bik,bjk->bijk', res_hs.view(pre_exc_1.shape[0], pre_exc_1.shape[1], -1), pre_exc_1.view(pre_exc_1.shape[0], pre_exc_1.shape[1], -1)).view(pre_exc_1.shape[0], pre_exc_1.shape[1] ** 2, pre_exc_1.shape[2], pre_exc_1.shape[3])
            cost_vol = torch.cat([cost_vol, entropy], 1)
            rnn_gate = self.circuit_gate(cost_vol).sigmoid()
            prev_hs = prev_hs * rnn_gate
        else:
            prev_hs = prev_hs * 0  # Dont allow the RNN on the first frame
        src_input = self.input_proj(src_search)
        hs, qs = self.featurefusion_network(src_template, mask_template, src_input, mask_search, pos_template[0], pos_search[0], exc=prev_hs)

        # Step 3, TD-FB incorporating hs
        pre_res_hs = qs.squeeze().view(1, self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2)  # .contiguous()

        # TD from Trans to Circuit
        # res_hs = self.cnl(self.circuit_tf_codeswitch(pre_res_hs))  # .contiguous())  # SUPER WEIRD but at traing time this op turns preres contiguous, but not at test time
        res_hs = self.cnl(self.circuit_tf_codeswitch(pre_res_hs)).contiguous()  # SUPER WEIRD but at traing time this op turns preres contiguous, but not at test time
        ##### This pooling is for technical reasons. Ideally dont have to do this but whatever...
        if 0:  # td_inh_1 is not None:
            from matplotlib import pyplot as plt
            f = plt.figure()
            plt.subplot(151);plt.imshow(ims.squeeze().permute(1, 2, 0).cpu());
            plt.subplot(152);plt.imshow((self.input_proj(src_search).squeeze()).mean(0).detach().cpu());
            plt.subplot(153);plt.title("Circuit-Transformer agreement", fontsize=6);plt.imshow((rnn_gate).squeeze().detach().cpu().mean(0))
            plt.subplot(154);plt.imshow(-(((prev_hs)).squeeze().mean(0).detach().cpu()));plt.title("Circuit Modulation", fontsize=6);
            plt.subplot(155);plt.title("Transformer", fontsize=6);plt.imshow((-hs.squeeze().view(1, self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2)).squeeze().mean(0).detach().cpu());  # plt.show()
            plt.savefig("test_gif/{}.jpg".format(self.gif_count))
            plt.close("all")
            self.gif_count += 1
            # plt.show()
            # print 
        else:
            self.gif_count = 0

        if td_inh_1 is None:
            td_inh_1 = self.cnl(self.circuit_td_inh_1_init(res_hs))
        exc_1, td_inh_1 = self.circuit_td_1(pre_exc_1, excitation=res_hs, inhibition=td_inh_1, activ=self.cnl)

        # Step 4 store hidden states
        self.exc_1 = exc_1
        self.inh_1 = inh_1
        self.td_inh_1 = td_inh_1
        self.res_hs = res_hs
        self.reset_hidden = False

        # Concat exc to hs too
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        import pdb;pdb.set_trace()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return_activities = info.get("return_activities", False)
        if return_activities:
            activities = {
                "transformer": -hs.squeeze().view(1, self.height, self.height, self.hidden_dim).permute(0, 3, 1, 2).squeeze().mean(0).detach().cpu(),
                "circuit": -prev_hs.squeeze().mean(0).detach().cpu()
            }
            return out, activities
        else:
            return out

    def template(self, z):
        if not isinstance(z, NestedTensor):
            z = nested_tensor_from_tensor_2(z)
        zf, pos_template = self.backbone(z)
        self.zf = zf
        self.pos_template = pos_template


class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, boxes=None, visible=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes, visible):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # pred_boxes needs to be associated with final target. This is largest distance between frames, for the Transformer.
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        if 'hgru_boxes' in outputs:
            # boxes is a N-length list of T timestep boxes for aux supervision
            circuit_boxes = outputs['hgru_boxes']
            # There are rare instances of an inf popping up in the scores. Clamp to control this. Unclear what the source is.
            # circuit_boxes = torch.clamp(circuit_boxes, 0, 1)

            """
            # Prepare labels by downsampling
            labels = boxes.permute(0, 2, 1, 3, 4)
            label_shape = labels.shape
            labels = labels.view(label_shape[0] * label_shape[2], label_shape[1], label_shape[3], label_shape[4])
            labels = F.interpolate(labels, circuit_boxes.shape[3:])
            labels = labels.view(label_shape[0], label_shape[1], label_shape[2], labels.shape[2], labels.shape[3])
            loss = nn.BCEWithLogitsLoss(reduction='none')
            # if 0:  # not torch.all(torch.isfinite(circuit_boxes)):
            #     print("Boxes are infinite")
            #     cb = torch.isinf(circuit_boxes).reshape(label_shape[0], -1).float()
            #     cb = cb.mean(-1)
            #     print("Num_infs: {} numvis: {}".format(cb, visible.mean(-1)))
            #     os._exit(1)
            # if 0:  # torch.any(torch.isnan(circuit_boxes)):
            #    print("Boxes are nan")
            #      cb = torch.isnan(circuit_boxes).reshape(label_shape[0], -1)
            #     cb = cb.mean(-1)
            #     print(cb)
            #     os._exit(1)
            # if 0:  # not torch.all(torch.isfinite(labels)) or torch.any(torch.isnan(labels)):
            #     print("Labels are infinite")
            #     os._exit(1)
            # if 0:  # torch.any(torch.isnan(labels)):
            #     print("Labels are nan")
            #     os._exit(1)
            bce = loss(circuit_boxes, labels)

            # loss_bbox = F.l1_loss(circuit_boxes, boxes, reduction='none')
            bce = bce * visible[:, None, :, None, None]
            bce = bce.sum((1, 2, 3, 4))
            nans = torch.isnan(bce)
            if torch.any(nans):
                print("NaNs detected in loss: {}".format(bce))
                nan_mask = 1 - nans.float()
                bce = bce[~nans]  # Filter the NaNs
                visible = visible * nan_mask[:, None]
            bce = bce.sum() / ((visible.sum() * labels.shape[2] * labels.shape[3]) + 1)
            losses['loss_giou_circuit'] = bce
            losses['loss_iou_circuit'] = torch.tensor(0.)
            """

            # boxes is a N-length list of T timestep boxes for aux supervision
            circuit_boxes = outputs['hgru_boxes']

            # loss_bbox = F.l1_loss(circuit_boxes, boxes, reduction='none')
            # Minimize dissimilarity between current and next bounding box
            cbx, cby, cv = [], [], []
            gious, ious = [], []
            frame_diff = 0
            start_frame = 3
            for bidx in range(start_frame, circuit_boxes.shape[1] - frame_diff):  # cb, bx, vis in zip(circuit_boxes, boxes, visible):
                cb = circuit_boxes[:, bidx]
                bx = boxes[:, bidx + frame_diff]  # Predictions
                vis = visible[:, bidx + frame_diff]
                cb = box_ops.box_cxcywh_to_xyxy(cb)
                bx = box_ops.box_cxcywh_to_xyxy(bx)
                if (cb[:, 2:] >= cb[:, :2]).all() and (bx[:, 2:] >= bx[:, :2]).all():
                    cbx.append(cb)
                    cby.append(bx)
                    cv.append(vis)
            if len(cbx) and len(cby) and len(cv):
                all_cb = torch.cat(cbx, 0)
                all_bx = torch.cat(cby, 0)
                all_vis = torch.cat(cv, 0)
                tgiou, tiou = box_ops.generalized_box_iou(
                    all_cb,
                    all_bx)
                cgiou = torch.diag(tgiou)
                ciou = torch.diag(tiou)
                cgious = cgiou * all_vis
                ciou = ciou.sum() / all_vis.sum()
                losses['loss_giou_circuit'] = (1 - cgiou).mean()
                losses['loss_iou_circuit'] = ciou
            else:
                losses['loss_giou_circuit'] = torch.tensor(0.)
                losses['loss_iou_circuit'] = torch.tensor(0.)


        # tgiou, tiou = box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(outputs['hgru_boxes'][idx]),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes))
        # cgiou = torch.diag(tgiou)
        # ciou = torch.diag(tiou)
        # losses['loss_giou_circuit'] = (1 - cgiou).sum() / num_boxes
        # losses['loss_iou_circuit'] = ciou.sum() / num_boxes

        # losses['loss_giou_circuit'] = torch.tensor(0)
        # losses['loss_iou_circuit'] = torch.tensor(0)

        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, boxes, visible):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, boxes, visible)

    def forward(self, outputs, targets, boxes, visible):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and "rnn" not in k and "hgru" not in k}

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)
        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos, boxes, visible))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    device = torch.device(settings.device)
    model.to(device)
    if settings.init_ckpt:
        print("Initializing from settings.init_ckpt")
        model = load_weights(model, settings.init_ckpt, strict=False)  # Not strict so we can add to the model
    return model

def transt_loss(settings):
    num_classes = 1
    matcher = build_matcher()
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    weight_dict['loss_giou_circuit'] = 0.5
    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion