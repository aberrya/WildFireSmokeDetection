# WildFire Smoke Detection 

### About
A wildfire is an uncontrolled fire. Every year, wildfire causes significant destruction of huge forest land, loss of animal and human lives, and wildlife food. Eearly detection of fire can significantly shorten the reaction time. The longer it takes to locate a fire, the harder it is to contain for fire staff. 

This is the submission for [lets-stop-wildfires-hackathon-2.0](https://aiformankind.org/lets-stop-wildfires-hackathon-2.0) to early detect WildFire smoke conducted by [AI For Mankind](https://aiformankind.org/) - a nonprofit organization.

Thanks to AIForManKind for providing [Quick Start Demo](https://github.com/aiformankind/wildfire-smoke-detection-camera) and providing labeled smoke Image Data Set. 

Also special thanks to [HPWREN](http://hpwren.ucsd.edu/) for providing access to HPWREN camera images.

### Saved Model
Submitted model is trained with EfficientDet-d3 using TensorFlow.

Data Set - 737 images. After augmenting (Horizontal Flip and brightness), dataset used was :-

   Training Images : 1739
   
   Validation Images : 111

Total training steps : 107000

### Fine Tuned Model
Saved model can be downloaded from https://drive.google.com/drive/folders/1R54ZCvD9-aNc-q59ZxUK_go9wO5qJKku?usp=sharingv
   

#### How to do Training and Inference

See [Model Training notebook](smoke_detection_model/notebooks/Model_Training_efficientdet_d3.ipynb) to do train model on smoke images.

For inference from saved model, refer to [inference notebook](smoke_detection_model/notebooks/smoke_detection_Inference_efficientdet-d3.ipynb)


### Resources

WildFire Resources
- [FUEGO Wildfire Detection Slides by Kinshuk Govil](https://tinyurl.com/rbrn4oq)
- [Wildland Fire Assessment System](https://journals.sagepub.com/doi/pdf/10.1155/2014/597368)
- [How Wildfire Works](https://science.howstuffworks.com/nature/natural-disasters/wildfire.htm/printable)
- [Wildland Fire: What is Hazard Fuel Reduction?](https://www.nps.gov/articles/what-is-hazard-fuel-reduction.htm)


Tensorflow Resources
- [Tensorflow Quickstart](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [TF Objection Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Object detection inference](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_from_saved_model_tf2_colab.ipynb)


Other Resources
- [Faster RCNN ResNnet](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a)
- [Train EfficientDet in TensorFlow](https://www.youtube.com/watch?v=yJg1FX2goCo)
- Data Augmentation using [roboflow](https://roboflow.com/)
- [Train object detection with Keras](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
- [Google Colab for training](https://colab.research.google.com/)

#### Others Model Tried

 - FatserRCNN ResNet101 - Got  best accuracy and lowest loss with this. But it was giving many False Positive for Fog images test.
  
 - Faster_rcnn_inception_resnet_v2_atrous_coco also gave good results for True Positives, but the prediction time is very high and it predicted many fog image as smoke.
 
#### Team members 
 [Anil](https://github.com/aberrya), [Khyati](https://github.com/khyati0502), [Krishna](https://github.com/Krrish3398) and Rama Revuri

#### Some inference results

![alt text](https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet_d3/True_Positives/1.png)
![alt text](https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet_d3/True_Positives/2.png)
![alt text](https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet_d3/True_Positives/4.png)
![alt text](https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet_d3/True_Positives/5.png)
![alt text](https://github.com/Krrish3398/WildFireSmokeDetection/blob/master/smoke_detection_model/results_efficientnet_d3/True_Positives/6.png)

