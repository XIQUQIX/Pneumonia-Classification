# Application of Chest X-Ray Images on Pneumonia Classification
## Abstract
This project explores deep learning techniques for classifying pneumonia using chest X-ray images. Initially, convolutional neural networks (CNNs) were employed to distinguish between normal and pneumonia cases. Subsequently, ResNet and DenseNet were utilized to classify bacterial versus viral pneumonia. Experiments were extended to a three-class problem (normal, bacterial, and viral), with the addition of GAN to enhance classification performance. While GAN slightly improved normal versus pneumonia classification, it did not significantly improve pneumonia type differentiation. The findings underscore the challenges of triple classification while validating CNN-based approaches for binary tasks.

## Overview
### What is the Problem?
This project addresses the automated classification of pneumonia from chest X-ray images. Specifically, the tasks include:
1.	Identifying whether a patient is normal or has pneumonia.
2.	Differentiating between bacterial and viral pneumonia.
3.	Expanding to triple classification: normal, bacterial pneumonia, or viral pneumonia.
### Why is this Problem Interesting?
Pneumonia remains a leading cause of death worldwide, particularly in children under five years in undeveloped countries. Rapid and accurate diagnosis can save lives, especially in resource-limited settings where radiologists are rare. Automated classification of chest X-rays using deep learning offers the potential to assist doctors, improve diagnostic accuracy, and expedite treatment decisions.

### What is the Approach?
We employed state-of-the-art deep learning models, including CNNs, ResNets, and DenseNets, for classification. Additionally, we incorporated a GAN-based framework to explore its utility in enhancing triple-class performance.
Rationale Behind the Approach
Deep learning models excel at image-based classification tasks. CNNs, ResNets, and DenseNets were chosen for their proven ability to capture image features. GANs were added to generate synthetic data and improve class separability, inspired by research showing GANs' potential in augmenting medical image datasets [4]. Our work builds on these techniques, integrating them into a comparative framework.

### Key Components of the Approach and Results
1.	Application for Dummy Class in Transfer Learning
Introduced a dummy class transfer learning. This approach was used to improve classification performance by providing a distinct class for the unclassified or ambiguous data, helping the model generalize better during training. We integrated this technique with ResNet and DenseNet, although its effectiveness was not as pronounced in the context of pneumonia classification.
2.	Use of Generative Adversarial Networks (GANs) for Data Augmentation
To address the data imbalance and enhance model performance, we leveraged GANs for data augmentation. GANs allowed us to generate synthetic pneumonia images, augmenting the existing dataset and improving the robustness of the models.
3.	Incorporation of a Classifier in the GAN Network
To extend the classification task to a three-class problem (normal, bacterial pneumonia, viral pneumonia), we integrated a classifier within the GAN architecture. The classifier was trained alongside the GAN to categorize images into one of the three categories.
4.	Application of Multiple Datasets
By combining these datasets, we were able to evaluate the models across various conditions, enabling a more comprehensive understanding of how the models perform under different scenarios. 

## Experiment Setup
### Dataset
We used two datasets from Kaggle:
1.	Chest X-Ray Images (Pneumonia) [1]
2.	Pediatric Pneumonia Chest X-Rays [2]
### Statistics
1. Images split into training (80%), validation (10%), and test (10%).
2. Class distribution: Normal, Bacterial Pneumonia, Viral Pneumonia.
### Implementation
1. Frameworks: PyTorch and TensorFlow for model implementation.
2.Models: CNN, ResNet, DenseNet, and GAN.
•	Hardware: Experiments were conducted on a GPU-enabled machine.
### Model Architectures
•	CNN: Sequential convolutional layers with ReLU activations and max pooling.
•	ResNet/DenseNet: Architectures with skip connections or dense blocks for deeper feature extraction.
•	GAN: Generator and discriminator networks trained adversarially to augment data.

## Experiment Results
### Main Results
1.	CNN achieved over 97% accuracy in binary classification (normal vs. pneumonia), which yielded high accuracy.
2.	ResNets and DenseNets performed well in distinguishing bacterial from viral pneumonia, and both achieved approximately 75% accuracy. 
3.	GANs improved synthetic data augmentation but did not significantly enhance triple-class results. Triple classification performance plateaued at around 70% accuracy.
### Supplementary Results
•	Parameter tuning revealed optimal learning rates and batch sizes for each model.
•	Data augmentation techniques do improve the transfer learning model a bit.
•	Applying a dummy class for classification job in ResNet proves meaningless.

## Discussion
### Insights
•	Binary classification tasks benefit significantly from deep learning models.
•	Differentiating bacterial from viral pneumonia is challenging due to subtle visual differences.
•	The GAN model needs to adjust a wide range of parameters and it is tough to converge. Thus, further funding is strongly required.

### Limitations
•	Class imbalance, too much bacterial pneumonia data with too little viral pneumonia data, in the dataset affected triple-class classification performance.
•	Difficulty in distinguishing bacterial from viral pneumonia due to overlapping radiological features.

### Future Work
•	Implement attention mechanisms to focus on region-specific features in X-rays.
•	Explore pre-training on larger datasets or additional augmentation techniques.
•	Collaborate with radiologists to integrate clinical context into the classification pipeline.

## Conclusion
This project demonstrates the efficacy of deep learning models for pneumonia classification using chest X-rays. While binary tasks achieve robust performance, multi-class classification remains challenging, highlighting opportunities for further research into feature extraction and augmentation.

## References
1.	Mooney, Paul Timothy. Chest X-ray Pneumonia. Kaggle, n.d. Accessed 28 Nov. 2024. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. 
2.	Mvd, Andrew. Pediatric Pneumonia Chest X-ray. Kaggle, n.d. Accessed 28 Nov. 2024. https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray.
3.	Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua (2014). Available at: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf. Proceedings of the International Conference on Neural Information Processing Systems (NIPS 2014). pp. 2672–2680.
4.	Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional Image Synthesis with Auxiliary Classifier GANs." arXiv preprint, arXiv:1610.09585v4 [stat.ML], 20 Jul. 2017. Available at: https://arxiv.org/abs/1610.09585.
