[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# Awesome Fine-Grained Classification

## Papers
- [Learning to Navigate for Fine-grained Classification] (https://arxiv.org/abs/1809.00287)
	
### Instance aware segmentation
- FCIS [https://arxiv.org/pdf/1611.07709.pdf]
	+ https://github.com/msracver/FCIS [MxNet]
- MNC [https://arxiv.org/pdf/1512.04412.pdf]
	+ https://github.com/daijifeng001/MNC [Caffe]
- DeepMask [https://arxiv.org/pdf/1506.06204.pdf]
	+ https://github.com/facebookresearch/deepmask [Torch]
- SharpMask [https://arxiv.org/pdf/1603.08695.pdf]
	+ https://github.com/facebookresearch/deepmask [Torch]
- Mask-RCNN [https://arxiv.org/pdf/1703.06870.pdf]
	+ https://github.com/CharlesShang/FastMaskRCNN [Tensorflow]
	+ https://github.com/jasjeetIM/Mask-RCNN [Caffe]
	+ https://github.com/TuSimple/mx-maskrcnn [MxNet]
	+ https://github.com/matterport/Mask_RCNN [Keras]
	+ https://github.com/facebookresearch/maskrcnn-benchmark [PyTorch]
	+ https://github.com/open-mmlab/mmdetection [PyTorch]
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
  + https://github.com/bernard24/RIS [Torch]
- FastMask [https://arxiv.org/pdf/1612.08843.pdf]
  + https://github.com/voidrank/FastMask [Caffe]
- BlitzNet [https://arxiv.org/pdf/1708.02813.pdf]
  + https://github.com/dvornikita/blitznet [Tensorflow]
- PANet [https://arxiv.org/pdf/1803.01534.pdf] [2018]
  + https://github.com/ShuLiu1993/PANet [Caffe]
- TernausNetV2 [https://arxiv.org/pdf/1806.00844.pdf] [2018]
	+ https://github.com/ternaus/TernausNetV2 [PyTorch]
- MS R-CNN [https://arxiv.org/pdf/1903.00241.pdf] [2019]
	+ https://github.com/zjhuang22/maskscoring_rcnn [PyTorch]

### Weakly-supervised segmentation
- SEC [https://arxiv.org/pdf/1603.06098.pdf]
  + https://github.com/kolesman/SEC [Caffe]

## RNN
- ReNet [https://arxiv.org/pdf/1505.00393.pdf]
  + https://github.com/fvisin/reseg [Lasagne]
- ReSeg [https://arxiv.org/pdf/1511.07053.pdf]
  + https://github.com/Wizaron/reseg-pytorch [PyTorch]
  + https://github.com/fvisin/reseg [Lasagne]
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
  + https://github.com/bernard24/RIS [Torch]
- CRF-RNN [http://www.robots.ox.ac.uk/%7Eszheng/papers/CRFasRNN.pdf]
  + https://github.com/martinkersner/train-CRF-RNN [Caffe]
  + https://github.com/torrvision/crfasrnn [Caffe]
  + https://github.com/NP-coder/CLPS1520Project [Tensorflow]
  + https://github.com/renmengye/rec-attend-public [Tensorflow]
  + https://github.com/sadeepj/crfasrnn_keras [Keras]
 
## GANS
- pix2pix [https://arxiv.org/pdf/1611.07004.pdf] [2018]
  + https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix [Pytorch]
  + https://github.com/affinelayer/pix2pix-tensorflow [Tensorflow]
- pix2pixHD [https://arxiv.org/pdf/1711.11585.pdf] [2018]
  + https://github.com/NVIDIA/pix2pixHD
- Probalistic Unet [https://arxiv.org/pdf/1806.05034.pdf] [2018]
  + https://github.com/SimonKohl/probabilistic_unet


## Graphical Models (CRF, MRF)
  + https://github.com/cvlab-epfl/densecrf
  + http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/
  + http://www.philkr.net/home/densecrf
  + http://graphics.stanford.edu/projects/densecrf/
  + https://github.com/amiltonwong/segmentation/blob/master/segmentation.ipynb
  + https://github.com/jliemansifry/super-simple-semantic-segmentation
  + http://users.cecs.anu.edu.au/~jdomke/JGMT/
  + https://www.quora.com/How-can-one-train-and-test-conditional-random-field-CRF-in-Python-on-our-own-training-testing-dataset
  + https://github.com/tpeng/python-crfsuite
  + https://github.com/chokkan/crfsuite
  + https://sites.google.com/site/zeppethefake/semantic-segmentation-crf-baseline
  + https://github.com/lucasb-eyer/pydensecrf

## Datasets:
  + [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
  + [Sift Flow Dataset](http://people.csail.mit.edu/celiu/SIFTflow/)
  + [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
  + [Microsoft COCO dataset](http://mscoco.org/)
  + [MSRC Dataset](http://research.microsoft.com/en-us/projects/objectclassrecognition/)
  + [LITS Liver Tumor Segmentation Dataset](https://competitions.codalab.org/competitions/15595)
  + [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)
  + [Pascal Context](http://www.cs.stanford.edu/~roozbeh/pascal-context/)
  + [Data from Games dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
  + [Human parsing dataset](https://github.com/lemondan/HumanParsing-Dataset)
  + [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  + [Microsoft AirSim](https://github.com/Microsoft/AirSim)
  + [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)
  + [COCO 2017 Stuff Segmentation Challenge](http://cocodataset.org/#stuff-challenge2017)
  + [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
  + [INRIA Annotations for Graz-02](http://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Daimler dataset](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
  + [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)
  + [INRIA Annotations for Graz-02 (IG02)](https://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Pratheepan Dataset](http://cs-chan.com/downloads_skin_dataset.html)
  + [Clothing Co-Parsing (CCP) Dataset](https://github.com/bearpaw/clothing-co-parsing)
  + [Inria Aerial Image](https://project.inria.fr/aerialimagelabeling/)
  + [ApolloScape](http://apolloscape.auto/scene.html)
  + [UrbanMapper3D](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&pm=14703)
  + [RoadDetector](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735)
  + [Cityscapes](https://www.cityscapes-dataset.com/)
  + [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

## Benchmarks
  + https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
  + https://github.com/meetshah1995/pytorch-semseg [PyTorch]
  + https://github.com/GeorgeSeif/Semantic-Segmentation-Suite [Tensorflow]
  + https://github.com/MSiam/TFSegmentation [Tensorflow]
  + https://github.com/CSAILVision/sceneparsing [Caffe+Matlab]
  + https://github.com/BloodAxe/segmentation-networks-benchmark [PyTorch]
  + https://github.com/warmspringwinds/pytorch-segmentation-detection [PyTorch]
  + https://github.com/ycszen/TorchSeg [PyTorch]
  + https://github.com/qubvel/segmentation_models [Keras]
  + https://github.com/qubvel/segmentation_models.pytorch [PyTorch]

## Evaluation code
  + [Cityscapes dataset] https://github.com/phillipi/pix2pix/tree/master/scripts/eval_cityscapes

## Starter code
  + https://github.com/mrgloom/keras-semantic-segmentation-example

## Annotation Tools:

  + https://github.com/AKSHAYUBHAT/ImageSegmentation
  + https://github.com/kyamagu/js-segment-annotator
  + https://github.com/CSAILVision/LabelMeAnnotationTool
  + https://github.com/seanbell/opensurfaces-segmentation-ui
  + https://github.com/lzx1413/labelImgPlus
  + https://github.com/wkentaro/labelme
  + https://github.com/labelbox/labelbox
  + https://github.com/Deep-Magic/COCO-Style-Dataset-Generator-GUI
  + https://github.com/Labelbox/Labelbox
  + https://github.com/opencv/cvat

## Results:

  + [MSRC-21](http://rodrigob.github.io/are_we_there_yet/build/semantic_labeling_datasets_results.html)
  + [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/)
  + [VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6)
  + https://paperswithcode.com/task/semantic-segmentation

## Metrics
  + https://github.com/martinkersner/py_img_seg_eval
  
## Losses
  + http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
  + https://arxiv.org/pdf/1705.08790.pdf
  + https://arxiv.org/pdf/1707.03237.pdf
  + http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
    
## Other lists
  + https://github.com/tangzhenyu/SemanticSegmentation_DL
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  + https://github.com/JackieZhangdx/InstanceSegmentationList
  
## Medical image segmentation:

- DIGITS
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/medical-imaging
  
- U-Net: Convolutional Networks for Biomedical Image Segmentation
  + http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
  + https://github.com/dmlc/mxnet/issues/1514
  + https://github.com/orobix/retina-unet
  + https://github.com/fvisin/reseg
  + https://github.com/yulequan/melanoma-recognition
  + http://www.andrewjanowczyk.com/use-case-1-nuclei-segmentation/
  + https://github.com/junyanz/MCILBoost
  + https://github.com/imlab-uiip/lung-segmentation-2d
  + https://github.com/scottykwok/cervix-roi-segmentation-by-unet
  + https://github.com/WeidiXie/cell_counting_v2
  + https://github.com/yandexdataschool/YSDA_deeplearning17/blob/master/Seminar6/Seminar%206%20-%20segmentation.ipynb
  
- Cascaded-FCN
  + https://github.com/IBBM/Cascaded-FCN
  
- Keras
  + https://github.com/jocicmarko/ultrasound-nerve-segmentation
  + https://github.com/EdwardTyantov/ultrasound-nerve-segmentation
  + https://github.com/intact-project/ild-cnn
  + https://github.com/scottykwok/cervix-roi-segmentation-by-unet
  + https://github.com/lishen/end2end-all-conv
  
- Tensorflow
  + https://github.com/imatge-upc/liverseg-2017-nipsws
  + https://github.com/DLTK/DLTK/tree/master/examples/applications/MRBrainS13_tissue_segmentation
  
- Using Convolutional Neural Networks (CNN) for Semantic Segmentation of Breast Cancer Lesions (BRCA)
  + https://github.com/ecobost/cnn4brca
  
- Papers:
  + https://www2.warwick.ac.uk/fac/sci/dcs/people/research/csrkbb/tmi2016_ks.pdf
  + Sliding window approach
	  - http://people.idsia.ch/~juergen/nips2012.pdf
  + https://github.com/albarqouni/Deep-Learning-for-Medical-Applications#segmentation
	  
 - Data:
   - https://luna16.grand-challenge.org/
   - https://camelyon16.grand-challenge.org/
   - https://github.com/beamandrew/medical-data
  
## Satellite images segmentation

  + https://github.com/mshivaprakash/sat-seg-thesis
  + https://github.com/KGPML/Hyperspectral
  + https://github.com/lopuhin/kaggle-dstl
  + https://github.com/mitmul/ssai
  + https://github.com/mitmul/ssai-cnn
  + https://github.com/azavea/raster-vision
  + https://github.com/nshaud/DeepNetsForEO
  + https://github.com/trailbehind/DeepOSM
  + https://github.com/mapbox/robosat
  + https://github.com/datapink/robosat.pink
  
 - Data:
  	+ https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-
	+ SpaceNet[https://spacenetchallenge.github.io/]
	+ https://github.com/chrieke/awesome-satellite-imagery-datasets

## Video segmentation

  + https://github.com/shelhamer/clockwork-fcn
  + https://github.com/JingchunCheng/Seg-with-SPN

## Autonomous driving

  + https://github.com/MarvinTeichmann/MultiNet
  + https://github.com/MarvinTeichmann/KittiSeg
  + https://github.com/vxy10/p5_VehicleDetection_Unet [Keras]
  + https://github.com/ndrplz/self-driving-car
  + https://github.com/mvirgo/MLND-Capstone
  + https://github.com/zhujun98/semantic_segmentation/tree/master/fcn8s_road
  + https://github.com/MaybeShewill-CV/lanenet-lane-detection

### Other

## Networks by framework (Older list)
- Keras
	+ https://github.com/gakarak/FCN_MSCOCO_Food_Segmentation
	+ https://github.com/abbypa/NNProject_DeepMask

- TensorFlow
	+ https://github.com/warmspringwinds/tf-image-segmentation
	
- Caffe
	+ https://github.com/xiaolonw/nips14_loc_seg_testonly
	+ https://github.com/naibaf7/caffe_neural_tool
	
- torch
	+ https://github.com/erogol/seg-torch
	+ https://github.com/phillipi/pix2pix
	
- MXNet
	+ https://github.com/itijyou/ademxapp

## Papers and Code (Older list)

- Simultaneous detection and segmentation

  + http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sds/
  + https://github.com/bharath272/sds_eccv2014
  
- Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation

  + https://github.com/HyeonwooNoh/DecoupledNet
  
- Learning to Propose Objects

  + http://vladlen.info/publications/learning-to-propose-objects/ 
  + https://github.com/philkr/lpo
  
- Nonparametric Scene Parsing via Label Transfer

  + http://people.csail.mit.edu/celiu/LabelTransfer/code.html
  
- Other
  + https://github.com/cvjena/cn24
  + http://lmb.informatik.uni-freiburg.de/resources/software.php
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation
  + http://jamie.shotton.org/work/code.html 
  + https://github.com/amueller/textonboost
  
## To look at
  + https://github.com/fchollet/keras/issues/6538
  + https://github.com/warmspringwinds/tensorflow_notes
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + https://github.com/desimone/segmentation-models
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + http://www.it-caesar.com/list-of-contemporary-semantic-segmentation-datasets/
  + https://github.com/MichaelXin/Awesome-Caffe#23-image-segmentation
  + https://github.com/warmspringwinds/pytorch-segmentation-detection
  + https://github.com/neuropoly/axondeepseg
  + https://github.com/petrochenko-pavel-a/segmentation_training_pipeline


## Blog posts, other:

  + https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
  + http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/
  + https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/binary-segmentation
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation
  + http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
  + https://medium.com/@barvinograd1/instance-embedding-instance-segmentation-without-proposals-31946a7c53e1

