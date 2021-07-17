# Object Identification and Localization Using Grad-CAM

## Introduction
What makes the network think the image should belong to a specific class? 
Grad-cam is a method for producing heatmaps that is applied to an already-trained neural network after training is complete and the parameters are fixed.
In this project, Grad-CAM is used to weakly localize ID cards and looking for the ROI in input images.

## Steps
In order to use class activation map, first, a pre-trained CNN is used to classify images that have ID cards. after that, the class activation map is computed for the last convolutional layer of the CNN.
it should result in something like this: 

![](data/original_image.jpg)
![](data/1.png)

The next step is to interpolate the class activation map image to become the same size as the original image.

![](./data/4.png)

Then the class activation map is converted to a binary image to find contour.

![](./data/binary_gradcam.png)

![](./data/found_contour.png)

Finally, a bounding box is drawn around the found contour. if there are multiple id cards in the input image, all of them will be localized.

![](./data/bouding_box.png)


#### More Visual Examples obtained by Xception trained on imagenet dataset:
|  Image | GradCAM  | Binarized GradCam  | Interpolated GradCam | Localized Object |
| -------|----------|------------|------------|------------|
|  ![](./data/2.jpg) | ![](./data/1.png)     |  ![](./data/3.png)    | ![](./data/4.png)   | ![](./data/5.png)   |
|  ![](./data/2_2.jpg) | ![](./data/2_1.png)     |  ![](./data/2_3.png)    | ![](./data/2_4.png)   | ![](./data/2_5.png)   |
|  ![](./data/3_1.jpg) | ![](./data/3_3.png)     |  ![](./data/3_2.png)    | ![](./data/3_3.png)    | ![](./data/3_5.png)  ![](./data/3_4.png)  |
|  ![](./data/4.jpg) | ![](./data/4_1.png)     |  ![](./data/4_2.png)    | ![](./data/4_3.png)    | ![](./data/4_4.png) |

This is the first phase of my final project of Deep Learning course taught by Dr.Mohammadi at IUST.

## Ref

[Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/)
