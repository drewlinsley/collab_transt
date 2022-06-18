import torch
from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt_circuit_encoder_test_mult as transt_models
# import ltr.models.tracking.transt_control as transt_models

from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU


def run(settings):
    # Most common settings are assigned in the settings struct
    debug = False
    if debug:
        settings.batch_size = 2  # 8  # 4  # 120  # 70 # 38
        settings.num_workers = 0  # 24  # 30  # 10  # 35  # 30 min(settings.batch_size, 16)
        settings.multi_gpu = False  # True  # True  # True #  True  # True  # True  # True

    else:
        settings.batch_size = 16  # 8  # 4  # 120  # 70 # 38
        settings.num_workers = 16  # 24  # 30  # 10  # 35  # 30 min(settings.batch_size, 16)
        settings.multi_gpu = True  # True  # True  # True #  True  # True  # True  # True

    settings.device = 'cuda'
    settings.description = 'TransT with default settings.'
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4.  # 4.0
    settings.template_area_factor = 2.
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3.0, 'template': 0}  # 3
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}  # 0.25

    settings.sequence_length = 10  # 20  # 64 NEXT  # Same as PT
    settings.rand = True  # If True Linear interpolate across 2 center/scale jitters. If False each frame is jittered.
    settings.occlusion = False
    settings.frame_multiplier = 6
    # settings.search_gap = 1  # Depreciated
    settings.pin_memory = False
    settings.init_ckpt = "pytracking/networks/transt.pth"

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    settings.move_data_to_gpu = True

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')  # votval
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    # coco_train = MSCOCOSeq(settings.env.coco_dir)

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    # The augmentation transform applied to the training set (individually to each image in the pair)
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    #                                 tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    # tfm.RandomHorizontalFlip(),
                                    # tfm.RandomAffine(p_flip=0.5, max_scale=1.5),
                                    # tfm.RandomBlur(1),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      joint=False,  # Whether or not to apply same transform to every image
                                                      transform=transform_train,
                                                      rand=settings.rand,
                                                      occlusion=settings.occlusion,
                                                      label_function_params=None,  # settings.label_function_params,
                                                      joint_transform=transform_joint)

    # The sampler for training
    # dataset_train = sampler.TransTSampler([got10k_train], [1], samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train, num_search_frames=settings.sequence_length, frame_sample_mode="rnn_interval")
    dataset_train = sampler.TransTSampler([lasot_train, got10k_train, trackingnet_train], [1, 1, 0], samples_per_epoch=1000*settings.batch_size, max_gap=settings.sequence_length * settings.frame_multiplier, processing=data_processing_train, num_search_frames=settings.sequence_length, frame_sample_mode="rnn_interval")
    # dataset_train = sampler.TransTSampler([lasot_train, got10k_train, trackingnet_train], [1, 1, 1], samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train, num_search_frames=settings.sequence_length, frame_sample_mode="interval")

    # dataset_train = sampler.TransTSampler([lasot_train, got10k_train, trackingnet_train], [1,1,1], samples_per_epoch=1000*settings.batch_size, max_gap=100, processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0, pin_memory=settings.pin_memory)  # settings.move_data_to_gpu == False)

    # Create network and actor
    model = transt_models.transt_resnet50(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    actor = actors.OldCircuitTranstActor(net=model, objective=objective)
    # actor = actors.CircuitTranstActor(net=model, objective=objective)

    # Optimizer
    # Change learning rate forthe Q we have changed and the RNN and the readout
    #         q = self.mix_q(torch.cat([q, self.mix_norm(exc)], -1))
    #        self.class_embed_new = MLP(hidden_dim * 2, hidden_dim, num_classes + 1, 3)
    #         self.bbox_embed_new = MLP(hidden_dim * 2, hidden_dim, 4, 3)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "circuit" in n or "rnn" in n or "trans" in n],
            "lr": 1e-4
        },
        {
            "params": [p for n, p in model.named_parameters() if "decoder" in n or "class_embed" in n or "bbox_embed" in n or "encoder" in n],
            "lr": 1e-6
        },
    ]
    print("Higher lr:")
    print([n for n, p in model.named_parameters() if "circuit" in n or "rnn" in n or "trans" in n])  # if "encoder" not in n and "input_proj" not in n and "decoder" not in n and "backbone" not in n and "embed" not in n])
    print("*" * 60)

    for n, p in model.named_parameters():
        if "circuit" in n or "rnn" in n or "trans" in n or "decoder" in n or "class_embed" in n or "bbox_embed" in n or "encoder" in n:
            print("TRAINING {}".format(n))
        else:
            p.requires_grad = False
            print("Removing grad on {}".format(n))
    optimizer = torch.optim.AdamW(param_dicts,
                                  weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(1000, load_latest=True, fail_safe=True)


