# WildFireSmokeDetection 
Submission for hackathon conducted by AI For Mankind (https://aiformankind.org/) to early detect WildFire smoke . See details of hackathon at https://aiformankind.org/lets-stop-wildfires-hackathon-2.0/


Thanks to AIForManKind for providing Quick Start Demo https://github.com/aiformankind/wildfire-smoke-detection-camera
and providning label Image Data Set.

### Saved Model
Submitted fined tune model is trained with EfficientDet-d3 using TensorFlow.

Data Set - 737 images. After augmenting (Horizontal Flip and added brightness), dataset was :-

   Training Images : 1739
   
   Validation Images : 111

Total training steps : 107000

### Fine Tuned Model
Saved model can be downloaded from https://drive.google.com/drive/folders/1R54ZCvD9-aNc-q59ZxUK_go9wO5qJKku?usp=sharingv
   

#### How to do Training and Inference

See [Model Training notebook](smoke_detection_model/notebooks/Model_Training_efficientdet_d3.ipynb) to do train youe model on smoke images.

For doing inference from saved model refer to [inference notebook](smoke_detection_model/notebooks/smoke_detection_Inference_efficientdet-d3.ipynb)


### Resources

WildFire Resources
- [FUEGO Wildfire Detection Slides by Kinshuk Govil](https://tinyurl.com/rbrn4oq)
- [Wildland Fire Assessment System] (https://journals.sagepub.com/doi/pdf/10.1155/2014/597368)
- [How Wildfire Works] (https://science.howstuffworks.com/nature/natural-disasters/wildfire.htm/printable)


Tensorflow Resources
- [Tensorflow Quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [TF Objection Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Object detection inference] (https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb)
- 

Other Resources
- [Faster RCNN ResNnet](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)
- [Train EfficientDet in TensorFlow](https://www.youtube.com/watch?v=yJg1FX2goCo)
- Data Augmentation using [roboflow](https://roboflow.com/)
- [Train object detection with Keras](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)

#### Others model Tried


 - YOLOV5 - With YOLOV5 with our dataset full smoke images were not detected properly.
 
 - SSD Mobile to solve this problem, but in results we found some limitations with some pattern of images. Training was very slow
 
 - FatserRCNN ResNet101 - Got very best accuracy and lowest loss with this. But it was giving many False Positive for Fog images test.
  
 - Faster_rcnn_inception_resnet_v2_atrous_coco gives good results for true positives but the prediction time is very high and do not solve False Postives                  problem(predicting fog as smoke) 
 
 - Segmentation part of this problem is also tried with Detectron2 model by preparing data from Labelme and then converted it to COCO with labelme2coco.py.
 
 - AP factor for segmenatation part was very less, so we are not including with our results.

#### Results
![alt text] (https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet/True%20Positives/1.png)
![alt text] (https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet/True%20Positives/2.png)
![alt text] (https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet/True%20Positives/3.png)
![alt text] (https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet/True%20Positives/4.png)
![alt text] (https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet/True%20Positives/5.png)

