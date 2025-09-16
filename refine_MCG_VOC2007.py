# # --------------------------------------------------------
# # Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# # Nvidia Source Code License-NC
# # --------------------------------------------------------
# from __future__ import absolute_import, division, print_function, unicode_literals
#
# from collections import OrderedDict
#
# import torch.nn as nn
# from wetectron.modeling import registry
# from wetectron.modeling.poolers import Pooler
#
# import torch
# import torchvision.ops
# from torch import nn
# import math
#
#
# # 可变形卷积
# class DeformableConv2d(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=1):
#         super(DeformableConv2d, self).__init__()
#         # 输入通道 输出通道 卷积核尺寸
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride if type(stride) == tuple else (stride, stride)
#         self.padding = padding
#
#         # init weight and bias
#         # 初始化参数
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
#         # 初始化偏置
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#
#         # offset conv
#         self.conv_offset_mask = nn.Conv2d(in_channels,
#                                           3 * kernel_size * kernel_size,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           padding=self.padding,
#                                           bias=True)
#
#         # init
#         self.reset_parameters()
#         self._init_weight()
#
#     #
#     def reset_parameters(self):
#         n = self.in_channels * (self.kernel_size ** 2)
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.zero_()
#
#     # 初始化权重
#     def _init_weight(self):
#         # init offset_mask conv
#         nn.init.constant_(self.conv_offset_mask.weight, 0.)
#         nn.init.constant_(self.conv_offset_mask.bias, 0.)
#
#     def forward(self, x):
#         out = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         x = torchvision.ops.deform_conv2d(input=x,
#                                           offset=offset,
#                                           weight=self.weight,
#                                           bias=self.bias,
#                                           padding=self.padding,
#                                           mask=mask,
#                                           stride=self.stride)
#         return x
#
#
# # to auto-load imagenet pre-trainied weights
# class Identity(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#
#     def forward(self, x):
#         return x
#
#
# class VGG_Base(nn.Module):
#     def __init__(self, features, cfg, init_weights=True):
#         super(VGG_Base, self).__init__()
#         self.features = features
#         if init_weights:
#             self._initialize_weights()
#         self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)
#
#     def forward(self, x):
#         x = self.features(x)
#         return [x]
#
#     def _initialize_weights(self):
#         for m in self.modules():
#
#             if isinstance(m, nn.Conv2d):
#                 print("see pth info", m)
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 print("see pth info", m)
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _freeze_backbone(self, freeze_at):
#         if freeze_at < 0:
#             return
#         assert freeze_at in [1, 2, 3, 4, 5]
#         layer_index = [5, 10, 17, 23, 29]
#         for layer in range(layer_index[freeze_at - 1]):
#             print("see", layer)
#             for p in self.features[layer].parameters():
#                 print("***/", self.features[layer])
#                 total=sum([param.nelement() for param in self.features[layer].parameters()])
#                 print(total)
#                 print(layer)
#                 p.requires_grad = False
#
#
#
#
# def make_layers(cfg, dim_in=3, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'I':
#             layers += [Identity()]
#         elif v == 'dcn':
#             dcnmodel = DeformableConv2d(512, 512)
#             layers += [dcnmodel, nn.BatchNorm2d(512)]
#         # following OICR paper, make conv5_x layers to have dilation=2
#         elif isinstance(v, str) and '-D' in v:
#             _v = int(v.split('-')[0])
#             conv2d = nn.Conv2d(in_channels, _v, kernel_size=3, padding=2, dilation=2)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(_v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = _v
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     # remove the last relu
#     return nn.Sequential(*layers[:-1])
#
#
# vgg_cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
#     'VGG16': [64, 64, 'M', 128, 128,
#               'M', 256, 256, 256,
#               'M', 512, 512, 512,
#               'M', 512, 512, 512],
#
#     'VGG16-OICR': [64, 64, 'M', 128, 128,
#                    'M', 256, 256, 256,
#                    'M', 512, 512, 512,
#                    'I', 'dcn', '512-D', '512-D'],
#
#     'E': [64, 64, 'M', 128, 128,
#           'M', 256, 256, 256, 256,
#           'M', 512, 512, 512, 512,
#           'M', 512, 512, 512, 512],
# }
#
#
# @registry.BACKBONES.register("VGG16")
# @registry.BACKBONES.register("VGG16-OICR")
# def add_conv_body(cfg, dim_in=3):
#     archi_name = cfg.MODEL.BACKBONE.CONV_BODY
#     body = VGG_Base(make_layers(vgg_cfg[archi_name], dim_in), cfg)
#     model = nn.Sequential(OrderedDict([("body", body)]))
#     model.out_channels = 512
#     return model
#
#
# @registry.ROI_BOX_FEATURE_EXTRACTORS.register("VGG16.roi_head")
# class VGG16FC67ROIFeatureExtractor(nn.Module):
#     def __init__(self, config, in_channels, init_weights=True):
#         super(VGG16FC67ROIFeatureExtractor, self).__init__()
#         assert in_channels == 512
#         resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
#         scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
#         sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
#         pooler = Pooler(
#             output_size=(resolution, resolution),
#             scales=scales,
#             sampling_ratio=sampling_ratio,
#         )
#         self.pooler = pooler
#
#         self.classifier = nn.Sequential(
#             Identity(),
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout()
#         )
#         self.out_channels = 4096
#
#         if init_weights:
#             self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, proposals):
#         # also pool featurs of multiple images into one huge ROI tensor
#         x = self.pooler(x, proposals)
#         x = x.view(x.shape[0], -1)
#         x = self.classifier(x)
#         return x
#
#     def forward_pooler(self, x, proposals):
#         x = self.pooler(x, proposals)
#         return x
#
#     def forward_neck(self, x):
#         x = x.view(x.shape[0], -1)
#         x = self.classifier(x)
#         return x
#
#
# from wetectron.config import cfg
#
# my_model = VGG_Base(make_layers(vgg_cfg["VGG16-OICR"], 3), cfg)
# i = 1
# print("****************************-----------------------")
# print(my_model)
# # for index in my_model.features.parameters():
# #     print(i, index.requires_grad)
# #     i = i + 1
# # for index in my_model.features:
# #     print(index)
#
#


import os
import pickle
import time
import cv2
import numpy as np
import torchvision
import torch
from model import ModifiedResNet
from models.clip.clip import _MODELS, _download, available_models
from torchvision.models._utils import IntermediateLayerGetter
from fenleiqi import Classifier
import xml.etree.ElementTree as ET


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias




with open(os.path.join("/home/sa/tr/couple_test_mix_2/VOC2007MCGproposal/"
                       "MCG-voc07_trainval.pkl"), 'rb') as f:
    proposals = pickle.load(f, encoding='latin1')
proposal_dict = {"boxes":[], "scores":[], "indexes":[]}
temp_i = 0
for index in range(len(proposals["boxes"])):
    print("begin {} proposal".format(index))
    # start_time = time.time()

    image = cv2.imread("/home/sa/tr/couple_test_mix_2/wetectron-master/datasets/voc/VOC2007/"
                       "JPEGImages/" +
                       "{:0>6}".format(proposals["indexes"][index])
                        + ".jpg")
    # print(str(proposals["indexes"][index])[0:4])
    # print(str(proposals["indexes"][index])[4:])
    # image = cv2.imread("/home/sa/tr/VOC2007/JPEGImages/" +
    #                     str(proposals["indexes"][index])[0:4] + "_" +
    #                     str(proposals["indexes"][index])[4:] + ".jpg")
    # print("/home/sa/tr/VOC2012/JPEGImages/" +
    #        proposals["indexes"][index][0:4] + "_" +
    #        proposals["indexes"][index][4:] + ".jpg")
    # print("/home/sa/tr/couple_test_mix_2/"
    #                    "wetectron-master/datasets/voc/"
    #                    "VOC2007/JPEGImages/" +
    #                    "{:0>6}".format(proposals["indexes"][index])
    #                     + ".jpg")

    data_deal = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
        torchvision.transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
#
    image_deal = data_deal(image)
    image_tensor = torch.unsqueeze(image_deal, dim=0).cuda()
    # image = image.cuda()
    model_path = _download(_MODELS["RN50"], os.path.expanduser("~/.cache/clip"))

    with open(model_path, 'rb') as opened_file:
        model = torch.jit.load(opened_file, map_location="cpu")
        state_dict = model.state_dict()

    counts: list = [len(set(k.split(".")[2] for k in state_dict
                            if k.startswith(f"visual.layer{b}")))
                            for b in [1, 2, 3, 4]]
    # vision_layers: (3, 4, 6, 3)
    vision_layers = tuple(counts)
    # vision_width: 64
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    vision_patch_size = None
    # output_dim:1024
    # embed_dim:2048
    output_dim, embed_dim = state_dict['visual.attnpool.c_proj.weight'].shape
    # vision_heads 32
    vision_heads = vision_width * 32 // 64
    # image_resolution: 224
    image_resolution = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5) * 32
    new_state_dict = dict()
    print("----------------")
    backbone = ModifiedResNet(layers=vision_layers,
                              output_dim=output_dim,
                              heads=vision_heads,
                              input_resolution=image_resolution,
                              width=vision_width,
                              bn=FrozenBatchNorm2d)
    backbone = backbone.cuda()
    num_channels = embed_dim
    new_state_dict.update({k.replace('visual.', ''): v
                           for k, v in state_dict.items()
                           if k.startswith('visual.')})

    region_prompt_path = "/home/sa/tr/couple_test_mix_2/" \
                         "wetectron-master/region_prompt_R50.pth"

    if region_prompt_path:
        region_prompt = torch.load(region_prompt_path, map_location='cpu')
        new_state_dict.update(region_prompt)
    backbone.load_state_dict(new_state_dict)  # load trained pth file
    # freeze backbone pth
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(False)
    return_layers = {'layer3': "layer3"}
    strides = [16]
    num_channels = [num_channels // 2]
    body = IntermediateLayerGetter(backbone, return_layers=return_layers).cuda()
    print("=============================")
    immer_feature_map = body(image_tensor)["layer3"]
    print(immer_feature_map.size())
    print("=============================")


    ##################################text encoder################################
    new_rois_tensor = torch.tensor([proposals["boxes"][int(index)]], dtype=torch.float32)
    new_rois_tensor = new_rois_tensor.cuda()
    new_rois_tensor_list = []
    new_rois_tensor_list.append(torch.squeeze(new_rois_tensor))
    roi_features = torchvision.ops.roi_align(
        immer_feature_map,
        new_rois_tensor_list,
        output_size=(14, 14),  # (reso, reso) if extra_conv else (reso // 2, reso // 2),
        spatial_scale=1 / 16,
        aligned=True)  # (bs * num_queries, c, 14, 14)
    roi_features = backbone.layer4(roi_features)
    roi_features = backbone.attnpool(roi_features)
    print("see see roi size():", roi_features.size())
    classifier = Classifier("clip_RN50", token_len=77, classifier_cache="").cuda()
    classifier = classifier.cuda()
    category_list = ["background"]


    base_anntation_path = "/home/sa/tr/couple_test_mix_2/wetectron-master/" \
                          "datasets/voc/VOC2007/Annotations/" \
                          + "{:0>6}".format(proposals["indexes"][index]) + ".xml"

    print(base_anntation_path)
    anno = ET.parse(base_anntation_path).getroot()
    for obj in anno.iter("object"):
        name = obj.find("name").text.lower().strip()
        if name not in category_list:
            category_list.append(name)

    print("category_list:", category_list)
    text_feature = classifier(category_list)
    # print("text_feature_size():", text_feature.size())
    similarity = roi_features @ text_feature.t()
    # print("similarity:", similarity.size())
    # end = time.time()
    # print(end-start_time)
    # print(similarity)
    p_i_c_i = np.argwhere(np.array(similarity.cpu())>=0.4)
    pro_index = [row[0] for row in p_i_c_i]
    print("type(pro_index)", type(pro_index))
    new_pro_index = []

    for i in pro_index:
        if new_pro_index.count(i) < 1:
            new_pro_index.append(i)
    print(new_pro_index)
    # image = cv2.imread("/home/sa/tr/couple_test_mix_2/"
    #                    "wetectron-master/datasets/voc/"
    #                    "VOC2007/JPEGImages/" +
    #                    "{:0>6}".format(proposals["indexes"][index])
    #                    + ".jpg")
    proposal_dict_boxes = np.zeros((len(new_pro_index), 4))
    proposal_dict_scores = np.zeros(len(new_pro_index))

    for filter in range(len(new_pro_index)):
        box = proposals["boxes"][index][new_pro_index[filter]]
        if (int(box[2])-int(box[0]))*(box[3]-box[1]) > 32*32:# remove small object
            image = cv2.rectangle(
                    image, tuple((int(box[0]), int(box[1]))),
                    tuple((int(box[2]), int(box[3]))), tuple((0, 255, 0)), 2)
            proposal_dict_boxes[filter] = np.array([[int(box[0]), int(box[1]),
                                                     int(box[2]), int(box[3])]])
            proposal_dict_scores[filter] = proposals["scores"][index][new_pro_index[filter]]
            # proposal_dict_scores[0][filter] = np.array([1], dtype=float)

    cv2.imwrite("/home/sa/tr/refine_proposal/" +
                str(proposals["indexes"][index])[0:4] + "_" +
                str(proposals["indexes"][index])[4:] + ".jpg", image)
    proposal_dict["boxes"].append(proposal_dict_boxes)
    proposal_dict["scores"].append(proposal_dict_scores)
    proposal_dict["indexes"].append(proposals["indexes"][index])
    category_list.clear()
    temp_i = temp_i + 1
    if temp_i > 5:
        with open('my_ss_proposal_2007_MCG.pkl', 'wb') as f:
            pickle.dump(proposal_dict, f)


with open('VOC2007MCGproposal/voc2007-MCG-trainval.pkl', 'wb') as f:
      pickle.dump(proposal_dict, f)

