import os
import sys
# 这个地方怎么改下，路径问题
sys.path.append(r'e:\PdCVSIG\github\iann\iann')

from util.exp_imports.default import *
from util.exp import load_config_file
from train.engine.trainer import ISTrainer
from data.datasets import ImagesDirDataset


ymload = load_config_file(r'E:\PdCVSIG\github\iann\iann\train\train_script\diy_script\train_config.yaml')

MODEL_NAME = ymload['model']['name']

loss_list = {
    'SigmoidBinaryCrossEntropyLoss': SigmoidBinaryCrossEntropyLoss(),
    'NormalizedFocalLossSigmoid': NormalizedFocalLossSigmoid(alpha=0.5, gamma=2),
}

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = ymload['dataset']['train_set']['crop_size']
    model_cfg.num_max_points = ymload['point']['num_max_points']

    if ymload['model']['type'] == 'hrnet':
        model = HRNetModel(
            width=ymload['model']['width'], 
            ocr_width=ymload['model']['ocr_width'],
            small=ymload['model']['small'],
            backbone_lr_mult=ymload['model']['backbone_lr_mult'],
            with_aux_output=ymload['model']['with_aux_output'], 
            use_leaky_relu=ymload['model']['use_leaky_relu'],
            use_rgb_conv=ymload['model']['use_rgb_conv'], 
            use_disks=ymload['model']['use_disks'], 
            norm_radius=ymload['model']['norm_radius']
        )
    else:
        model = DeeplabModel(
            backbone=ymload['model']['backbone'],
            deeplab_ch=ymload['model']['deeplab_ch'],
            aspp_dropout=ymload['model']['aspp_dropout'],
            backbone_norm_layer=ymload['model']['backbone_norm_layer'],
            backbone_lr_mult=ymload['model']['backbone_lr_mult'],
            with_aux_output=ymload['model']['with_aux_output'], 
            use_leaky_relu=ymload['model']['use_leaky_relu'],
            use_rgb_conv=ymload['model']['use_rgb_conv'], 
            use_disks=ymload['model']['use_disks'], 
            norm_radius=ymload['model']['norm_radius']
        )

    # model.to(cfg.device)  # 没找到device在哪儿设置的
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = ymload['dataset']['train_set']['batch_size']
    cfg.val_batch_size = ymload['dataset']['val_set']['batch_size']
    crop_size = ymload['dataset']['train_set']['crop_size']

    loss_cfg = edict()
    loss_cfg.instance_loss = loss_list[ymload['loss']['instance_loss']]
    loss_cfg.instance_loss_weight = ymload['loss']['instance_loss_weight']
    loss_cfg.instance_aux_loss = loss_list[ymload['loss']['instance_aux_loss']]
    loss_cfg.instance_aux_loss_weight = ymload['loss']['instance_aux_loss_weight']

    # aug搞起来太多了
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=ymload['point']['prob_gamma'],
        merge_objects_prob=ymload['point']['merge_objects_prob'],
        max_num_merged_objects=ymload['point']['max_num_merged_objects']
    )

    trainset = ImagesDirDataset(
        os.path.join(ymload['dataset']['path'], ymload['dataset']['train_set']['split_name']),
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.0,
        points_sampler=points_sampler,
        epoch_len=12771
    )

    valset = ImagesDirDataset(
        os.path.join(ymload['dataset']['path'], ymload['dataset']['val_set']['split_name']),
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=126
    )

    optimizer_params = {
        'learning_rate': ymload['learning_rate']['base']
    }

    # lr_scheduler = partial(
    #     paddle.optimizer.lr.MultiStepDecay,
    #     milestones=ymload['learning_rate']['scheduler']['milestones'], 
    #     gamma=ymload['learning_rate']['scheduler']['gamma']
    # )

    trainer = ISTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        # lr_scheduler=lr_scheduler,
        checkpoint_interval=ymload['checkpoint_interval'],
        image_dump_interval=ymload['image_dump_interval'],
        metrics=[AdaptiveIoU()],
        max_interactive_points=model_cfg.num_max_points
    )
    
    trainer.run(num_epochs=ymload['num_epochs'])