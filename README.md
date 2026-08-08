# CR2T
CR2T: Region-Aware Prompting CLIP for Weakly Supervised Object Detection with Region Refinement and Test Time Tuning. 
Extensive experiments demonstrate CR2T gains performance improvements compared with baseline OICR, MIST, OD-WSCL and FI-WSOD on VOC 2012/2007 and COCO 2017 datasets. 
Meanwhile, CR2T achieves the state-of-the-art performance on VOC 2007/2012 and COCO 2017 datasets.

Thanks to the following for their contributions to the open source community.
OICR   https://arxiv.org/pdf/1704.00138
MIST   https://arxiv.org/pdf/2004.04725
OD-WSCL https://arxiv.org/pdf/2208.07576
FI-WSOD  https://ieeexplore.ieee.org/document/9854139

1. RAF -----》 filter region proposals
2. image+refined proposals ----》 WSOD + CCF
3. CCF with tta tunes parameters
