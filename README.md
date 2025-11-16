# Brain-Tumor-MRI-Classification-Using-Convolution-Neural-Network-CNN-Glioma-vs-Non-Glioma

# Project Description
This project( school learning project) implements a convolutional neural network (CNN) pipeline for classifying brain MRI images as glioma (positive) or non-glioma (negative). 
It uses Apache Spark for efficient, distributed preprocessing and large-scale data augmentation, addressing class imbalance and scalability for robust medical image analysis.

# Features
**Distributed Preprocessing**: Uses Apache Spark to handle large MRI datasets efficiently.
**Model Architecture**: Lightweight Keras CNN for binary image classification.
**Class Balancing**: Includes automatic computation of class weights.
**Image Augmentation**: Option for large-scale automated image transformations to improve model generalization.
**Evaluation Metrics**: Reports accuracy, precision, recall, F1-score, and confusion matrix on the test set.

# Requirements (Required Depencies)
Python 3.8 or higher
Apache Spark
TensorFlow and Keras
scikit-learn
matplotlib
Pillow
# Model Perfromance Metrics
Test Overall Accuracy: 92%
Precision: 0.90 (non-glioma), 0.94 (glioma)
Recall: 0.95 (non-glioma), 0.88 (glioma)
F1-Score: 0.92 (non-glioma), 0.91 (glioma)

# Limitations & Future Work
Image Resolution: The model is optimized for performance at 32×32 resolution due to limited computing resource 
Future improvements may include higher resolutions or alternative architectures.
Augmentation: Large-scale augmentation is planned for further robustness.
Evaluation: Additional metrics (such as Intersection-over-Union for segmentation) may be incorporated.

# License & Acknowledgments
Dataset provided by Figshare. (https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
# Author
Peter Mvuma Graduate Student MS Health Informatics
Email: pmvuma@mtu.edu
