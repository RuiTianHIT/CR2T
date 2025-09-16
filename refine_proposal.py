import os
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


point_class_list = ["aeroplane", "bird", "bottle",
                    "car", "chair", "diningtable",
                    "person", "tvmonitor"]

proposal_dict = {"boxes": [], "scores": [], "indexes": []}

temp_i = 0
with open(os.path.join("/home/sa/tr/OD-WSCL-master/proposal/SS/voc",
                       "SS-voc12_trainval.pkl"), 'rb') as f:
    # 加载候选区域的文件 proposals这个现在就是候选区域了
    proposals = pickle.load(f, encoding='latin1')
    for index in range(len(proposals["boxes"])):
        print("begin {} proposal".format(index))
        base_anntation_path = "/home/sa/tr/VOC2012/Annotations/" \
                              + str(proposals["indexes"][index])[0:4] \
                              + "_" + str(proposals["indexes"][index])[4:] + ".xml"
        # print(base_anntation_path)
        anno = ET.parse(base_anntation_path).getroot()
        # 得到了这个图像中存在什么类别
        proposal_list = []
        # 把这个图像中的所有的类别拿到，然后绝对这张照片的候选区域我是否进行精炼
        for obj in anno.iter("object"):
            name = obj.find("name").text.lower().strip()
            proposal_list.append(name)
        sub_flag = set(proposal_list).issubset(set(point_class_list))
        print(proposal_list)
        if sub_flag:
            print("wo kai shi refine")
            # 这种情况下才对此图像进行refine候选区域
            # 读取图像，将图像归一化
            image = cv2.imread("/home/sa/tr/VOC2012/JPEGImages/" +
                               str(proposals["indexes"][index])[0:4] + "_" +
                               str(proposals["indexes"][index])[4:] + ".jpg")
            data_deal = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
                torchvision.transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
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

            immer_feature_map = body(image_tensor)["layer3"]


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

            classifier = Classifier("clip_RN50", token_len=77, classifier_cache="").cuda()
            classifier = classifier.cuda()
            ##############
            category_list = ["background"]
            anno = ET.parse(base_anntation_path).getroot()
            for obj in anno.iter("object"):
                name = obj.find("name").text.lower().strip()
                if name not in category_list:
                    category_list.append(name)

            # print("category_list:", category_list)
            text_feature = classifier(category_list)
            # print("text_feature_size():", text_feature.size())
            similarity = roi_features @ text_feature.t()
            # print("similarity:", similarity.size())
            # end = time.time()
            # print(end-start_time)
            # print(similarity)
            p_i_c_i = np.argwhere(np.array(similarity.cpu()) >= 0.4)
            pro_index = [row[0] for row in p_i_c_i]
            # print("type(pro_index)", type(pro_index))
            new_pro_index = []

            for i in pro_index:
                if new_pro_index.count(i) < 1:
                    new_pro_index.append(i)
            print(new_pro_index)

            proposal_dict_boxes = np.zeros((len(new_pro_index), 4))
            proposal_dict_scores = np.ones((1, len(new_pro_index)))

            for filter in range(len(new_pro_index)):
                box = proposals["boxes"][index][new_pro_index[filter]]
                if (int(box[2]) - int(box[0])) * (box[3] - box[1]) > 32 * 32:  # remove small object
                    image = cv2.rectangle(
                        image, tuple((int(box[0]), int(box[1]))),
                        tuple((int(box[2]), int(box[3]))), tuple((0, 255, 0)), 2)
                    proposal_dict_boxes[filter] = np.array([[int(box[0]), int(box[1]),
                                                             int(box[2]), int(box[3])]])
                    proposal_dict_scores[0][filter] = np.array([1], dtype=float)

            cv2.imwrite("/home/sa/tr/refine_proposal/" +
                        str(proposals["indexes"][index])[0:4] + "_" +
                        str(proposals["indexes"][index])[4:] + ".jpg", image)
            print("******", len(proposal_dict_boxes))
            proposal_dict["boxes"].append(proposal_dict_boxes)
            proposal_dict["scores"].append(proposal_dict_scores)
            proposal_dict["indexes"].append(proposals["indexes"][index])
            category_list.clear()


        else:
            # 不refine候选区域，保留原始的区域
            print("不refine候选区域")
            print(len(proposals["boxes"][index]))
            proposal_dict["boxes"].append(proposals["boxes"][index])
            proposal_dict["scores"].append(proposals["scores"][index])
            proposal_dict["indexes"].append(proposals["indexes"][index])

        proposal_list.clear()



        # temp_i = temp_i + 1
        # if temp_i > 5:
        #     with open('my_ss_proposal_2012_1109.pkl', 'wb') as f:
        #         pickle.dump(proposal_dict, f)

with open('proposal_refine_2012/voc_my_proposal_trainval2012.pkl',
          'wb') as f:
      pickle.dump(proposal_dict, f)
