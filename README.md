# Blood-Cell-Classifier
Artificial neural network for image classification of different types of white blood cells.

## Overview

This repository contains my CNN model building project on the Blood Cells Image Dataset from [Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)
The aim is to build a robust model that can predict which type of normal peripheral blood cell is found in a microscopic image.

## Introduction


## Results & Discussion

** Description of the dataset and preprocessing**

The dataset comprises microscopic images measuring 360 x 360 pixels, depicting normal white blood cells classified into eight distinct categories: Neutrophils, Eosinophils, Basophils, Lymphocytes, Monocytes, Immature Granulocytes (IG), Erythroblasts, and Platelets/Thrombocytes. To prepare the data for model building, it was resized to 224 x 224 pixels, shuffled, and batched into groups of 32 images. Subsequently, a split of 80% training data and 20% test data was performed. Within the training set, an additional 20% was allocated as a validation set, resulting in a distribution of 64% training, 16% validation, and 20% test images. The pixel intensities ranged from 3.5 to 255, necessitating appropriate rescaling.
The dataset's class distribution is imbalanced, with Neutrophils representing the most prevalent class at 19.4% of images and Lymphocytes being the least common class at 7.1% [Fig. 1].

**Model 1 (Baseline DNN)**
Model 1 was designed with a simple architecture consisting of 8 layers arranged in a sequence (deep neural network; DNN). The architecture begins with a rescaling layer to standardize pixel values; then a flattening layer follows to convert image data into a one-dimensional format. Five hidden dense layers, each with 128 ReLU-activated neurons, are followed by the final output layer, which has eight neurons utilizing a softmax activation function to classify the eight types of white blood cells. The model achieved a validation accuracy of approximately 21.10%, which is only slightly better than that of a random model, which would be 19.4% according to the most frequent class [See Fig. 1]. Since the basic DNN is unable to capture structured data, a simple convolutional neural network (CNN) was tested.

** Model 2 (Simple CNN)**

Model 2 employes a basic convolutional neural network (CNN) design to tackle the structure present in image data. This model begins with the rescaling layer, followed by a Conv2D layer consisting of 8 filters to extract critical image features. Next, a MaxPool2D layer decreases spatial dimensions to enhance computational efficacy. A flattening layer prepares the data for the final output layer.
Unfortunately, Model 2 revealed pronounced overfitting during training, as evidenced by a training accuracy of approximately 97% and a noticeably lower validation accuracy of around 85%. To counter this overfitting, data augmentation techniques are applied to enhance the model's ability to generalize.

** Model 3 (CNN with data augmentation)**

Model 3 has a similar architecture to Model 2, except that a data augmenting layer precedes the Conv2D layer. The data augmentation layer applies random flips and rotations to the input images, increasing the diversity of the training data.
Model 3 benefitted from augmentation techniques, resulting in validation and training accuracies of approximately 88%. This demonstrates its proficient ability to overcome overfitting, enabling the development of a more complex CNN architecture.

**Model 4 (More complex CNN with data augmentation)**

Model 4 employs an advanced CNN architecture that incorporates rescaling and data augmentation. It consists of three convolution (Conv) blocks, each block consisting of a Conv2D layer and a MaxPool2D layer. The filter size gradually increases from 8 to 16 to 32 filters, using ReLU activation functions. The initial Conv block has a kernel size of 5x5 pixels, while all subsequent blocks have a kernel size of 3x3 pixels.
During the training period, the model showed a significant increase in performance by achieving a validation accuracy of approximately 94.6%, which is notably higher than that of the previous models. It is crucial to note that this improvement was attained without significant overfitting, which indicates the effectiveness of the more intricate CNN architecture in capturing complex patterns. 
Additionally, considering the notable variations in low-level features displayed in certain images within the dataset, such as the reddish granules in eosinophils, a shift to the ResNet50 architecture was implemented. This architecture is recognized for its exceptional ability to capture low-level feature variations.

** Model 5 and Model 6 (Transfer learning with ResNet50)**

Models 5 and 6 use the ResNet50 architecture for extracting features in image classification tasks, with some minor differences in the training methods. These include the incorporation of the ResNet50-layer after data augmentation and the addition of a GlobalAvgPool2D-layer between the ResNet50-layer and the output layer. In Model 5, the ResNet50 model serves as a feature extractor, and its layers are frozen. Model 5 achieved a validation accuracy of about 93%, which is comparable to Model 4. In Model 6 the entire ResNet50 base model is unfrozen, allowing all its layers to be trained. It demonstrated an exceptional performance of around 98% accuracy on the validation set, rendering it a top contender for the best model.

** Models' performance on the test set**

In the final test phase, all models underwent evaluation on the test dataset. Model 1 achieved an accuracy of 20%, representing the baseline performance. Model 2 exhibited a significant improvement, achieving an accuracy of 84%, while Model 3 continued this trend, reaching an accuracy of 87%. Model 4 showed substantial progress, achieving a test accuracy of 95%. The most remarkable performance was observed in Model 6, which utilized fine-tuning and the ResNet50 architecture resulting in an outstanding test accuracy of 98%.

** Errors of Model 6 on the test data**

To accurately evaluate Model 6's performance, it is crucial to identify the specific areas where it encountered difficulties. The class accuracies displayed variances, with some exhibiting ranges from 94.2% for the ig class to a flawless 100.0% for lymphocytes. On 29 occasions, Model 6 confused the ig category for neutrophils and 12 times it erroneously identified platelets as lymphocytes. Another prevalent error was the misclassification of neutrophils as ig, which happened 7 times. The majority of classes had high $F_1$-scores of 0.99, showcasing the strong classification capability of the model. However, the lymphocytes category had a slightly lower $F_1$-score of 0.95, revealing a specific area where Model 6 could benefit from further enhancements. It is worth noting that lymphocytes are the least commonly occurring class in the dataset, comprising only 7.1% of instances.
The analysis of 60 inaccurately predicted images discovered that numerous instances contain more than one cell type present in the image. For instance, in Figure XXX, the central cell bears resemblance to a neutrophil, while the cell on the right-hand side bears resemblance to a lymphocyte. Model 6 predicted that this instance is a lymphocyte, but the label indicated it as a neutrophil.
There is potential for improvement in implementing a semi-supervised learning approach. The first model should cluster images based on their "stained cell content" (i.e., the amount and intensity of bluish pixels in the image). This unsupervised model will detect outliers with more than one cell per image and only pass images with one cell to the classification model.
