
# Ayurvedic Plant Analysis using Deep Learning

## Overview
This repository contains the implementation of a deep learning-based approach for the automated recognition of Ayurvedic plant species. The project utilizes Convolutional Neural Networks (CNN) and MobileNetV2 architectures to achieve high accuracy in identifying medicinal plants from leaf images.

## Authors
- Priyanka Vemulakonda - [vemulakonda.priyanka@gmail.com](mailto:vemulakonda.priyanka@gmail.com)
- Syed Afreen - [afreensd07@gmail.com](mailto:afreensd07@gmail.com)
- Tanishq Rachamalla - [tanishqrachamalla12@gmail.com](mailto:tanishqrachamalla12@gmail.com)


## Abstract
Ayurvedic plants are rich sources of therapeutic compounds, yet accurately identifying these plants poses significant challenges. This project introduces a deep learning approach for the automated recognition of Ayurvedic plant species, combining specialized datasets to create a comprehensive collection of over 9000 images. The MobileNetV2 architecture, optimized through depthwise separable convolutions and transfer learning, achieves over 81% testing accuracy, surpassing the CNN model by 13%.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)

## Introduction
India is home to 40% of the world's medicinal plants, yet this wealth contrasts sharply with the limited utilization of this knowledge. Our initiative seeks to address this discrepancy by leveraging deep learning to automate the recognition of Ayurvedic plant species, thereby empowering our nation with in-depth knowledge about these plants.

## Dataset
The dataset used is a robust compilation from multiple sources, consisting of over 9000 images across 103 medicinal plant categories. It includes:
- **Indian Medicinal Plant Leaf Images**: 80 classes, over 5000 images with varied backgrounds.
- **General Medicinal Plant Leaf Images**: 30 classes, 2000 images with a uniform white background.

![Dataset](https://github.com/Tanishq251/A-Deep-Learning-Based-Plant-Analysis-for-Ayurvedic-Application/assets/104064377/cf3f54b4-16cf-41f2-a9ef-93864b5e5dca)


-[Training Data](https://drive.google.com/drive/folders/1XMSUxYfGTmbhDJS-mS2RizxQxFReQOH8?usp=drive_link)
-[Testing Data](https://drive.google.com/drive/folders/1V0QqESerk0qDFah9HPHiGe1Cyb-gNTgW?usp=drive_link)

## Methodology
### CNN Architecture
The Convolutional Neural Network (CNN) model used in this project is a sequential architecture composed of several layers designed to learn hierarchical features from the input images. The architecture includes multiple convolutional layers, max pooling layers, dropout layers to prevent overfitting, and dense layers for classification. The CNN model achieved a baseline accuracy of 68%.

![CNN Architecture](https://github.com/Tanishq251/A-Deep-Learning-Based-Plant-Analysis-for-Ayurvedic-Application/assets/104064377/9c4a7410-66cc-4ad7-8d3e-88423763b3e3)


### MobileNetV2 Architecture
The MobileNetV2 model utilized in this project is based on a pre-trained network on ImageNet, fine-tuned for our specific task of Ayurvedic plant species classification. The key features of MobileNetV2 include depthwise separable convolutions, bottleneck layers, and inverted residuals, which enhance efficiency and representational power. This architecture achieved an accuracy of 81%.


## Results
The experimental results demonstrate the superior performance of the MobileNetV2 architecture compared to the traditional CNN model. The MobileNetV2 model, leveraging depthwise separable convolutions and transfer learning, significantly outperformed the CNN model.

### Model Performance

The MobileNetV2 architecture achieved an accuracy of 81%, which is 13% higher than the CNN model. This performance boost underscores the efficacy of depthwise separable convolutions and transfer learning in handling complex image recognition tasks in Ayurvedic plant species identification.

![Model Performance](https://github.com/Tanishq251/A-Deep-Learning-Based-Plant-Analysis-for-Ayurvedic-Application/assets/104064377/d4dd61bc-ad0c-4cb2-b525-63357e74d27b)



`

