# Face Recognition using a pre trained deep learning model i.e., Caffe model and using SVM 
 - Create dataset of face images
 - - Detect faces using ```deploy.prototxt``` and ```res10_300x300_ssd_iter_140000.caffemodel```. (Learn more about [face detection]
 - Extract face embeddings for each face present in the image using pretrained [OpenFace](https://cmusatyalab.github.io/openface/) model ```openface_nn4.small2.v1.t7```. 
 - Train a SVM model on the face embeddings to recognize faces 

## Getting Started
How to use
```    
 - Create dataset of face images.
 - Place the face images in dataset folder.
 - Extract facial embeddings.
```python extract_embeddings.py```
 - Train the SVM model
```python train_model.py```
 - Test the model
```python recognize_video.py```

## Prerequisites
- Python 3.5
- OpenCV
```
sudo apt-get install python-opencv
```
