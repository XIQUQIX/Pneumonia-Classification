# Application of Chest X-Ray Images on Pneumonia Classification
## Project Overview
Teammates: Rongjia Sun, Churou Deng

## Environment Setup
### Required Library
torch (version == '2.4.1+cu118')

tensorflow (version == '2.18.0')

PIL

kagglehub

sklearn

seaborn

numpy

matplotlib

## How to use
1. Prepare an environment with Jupyter Notebook and above libraries
2. Download dataset in dataset file or at [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia] and [https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray]
3. Run CNN/CNN.ipynb or CNN/CNN_bin_classifier.ipynb to implement CNN for binary classification between normal and pneumonia on dataset chest-xray-pneumonia
4. Run Transfer_Learning/Transfer_Learning_dense.ipynb to implement denseNet for binary classification between bacterial and viral pneumonia on dataset chest-xray-pneumonia
5. Run Transfer_Learning/Transfer_Learning_dense_pediatric.ipynb to implement denseNet for binary classification between bacterial and viral pneumonia on dataset pediatric-pneumonia-chest-xray
6. Run Transfer_Learning/Transfer_Learning_res.ipynb to implement ResNet for binary classification between bacterial and viral pneumonia on both chest-xray-pneumonia and pediatric-pneumonia-chest-xray
7. Run GAN/GAN_tri_classifier.ipynb to implement GAN for triple classification among normal, bacterial pneumonia, and viral pneumonia on dataset chest-xray-pneumonia

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

Furthermore, while most publicly available implementations focus solely on binary classification, distinguishing between normal and pneumonia cases, our work goes a step further by exploring multi-class classification to differentiate between bacterial and viral pneumonia. This is an area with limited prior research or publicly available benchmarks, making our contributions particularly noteworthy in addressing this gap.

### What is the Approach?
We employed state-of-the-art deep learning models, including CNNs, ResNets, and DenseNets, for classification. Additionally, we incorporated a GAN-based framework to explore its utility in enhancing triple-class performance.
Rationale Behind the Approach.
Deep learning models excel at image-based classification tasks. CNNs, ResNets, and DenseNets were chosen for their proven ability to capture image features. GANs were added to generate synthetic data and improve class separability, inspired by research showing GANs' potential in augmenting medical image datasets [4]. Our work builds on these techniques, integrating them into a comparative framework.

### Key Components of the Approach and Results
1.	Application for Dummy Class in Transfer Learning
Introduced a dummy class transfer learning. This approach was used to improve classification performance by providing a distinct class for the unclassified or ambiguous data, helping the model generalize better during training. We integrated this technique with ResNet and DenseNet, although its effectiveness was not as pronounced in the context of pneumonia classification.
2.	Use of Generative Adversarial Networks (GANs) for Data Augmentation
To address the data imbalance and enhance model performance, we leveraged GANs for data augmentation. GANs allowed us to generate synthetic pneumonia images, augmenting the existing dataset and improving the robustness of the models. Despite promising applications in other domains, the GAN-generated data did not substantially improve the classification of pneumonia types (bacterial vs. viral), but it proved valuable in maintaining the balance between normal and pneumonia images.
3.	Incorporation of a Classifier in the GAN Network
To extend the classification task to a three-class problem (normal, bacterial pneumonia, viral pneumonia), we integrated a classifier within the GAN architecture. The classifier was trained alongside the GAN to categorize images into one of the three categories. While this approach balanced the overall performance, the distinction between bacterial and viral pneumonia remained challenging. This indicates that the task might require additional features beyond those captured in the current datasets or architectures.
4.	Application of Multiple Datasets
By combining these datasets, we were able to evaluate the models across various conditions, enabling a more comprehensive understanding of how the models perform under different scenarios.
5.  Improvements in CNN Training with Gradient Stabilization
While training CNNs for binary and triple classification tasks, we encountered the vanishing gradient problem, particularly in deeper networks. This issue hindered the training process and led to slow convergence. To address this, we employed techniques such as Batch Normalization, which normalizes layer inputs to stabilize the learning process, and ReLU activation functions, which mitigate gradient shrinkage by introducing non-linearity. These adjustments significantly improved the gradient flow and accelerated convergence, enabling the CNN models to better differentiate between normal, bacterial, and viral pneumonia. These methods contributed to achieving baseline performance before integrating more complex architectures and GAN-based augmentation.
6. Application of Transfer Learning and its advantages
Models using transfer learning outperformed those trained from scratch in terms of both classification speed and accuracy. In binary classification tasks, DenseNet and ResNet achieved accuracies of up to 75% with the aid of transfer learning, while reducing training time on the Pediatric Pneumonia dataset and mitigating overfitting caused by data scarcity. Furthermore, we fine-tuned the higher layers of DenseNet and ResNet to adapt the models to domain-specific features of chest X-rays, while freezing lower layers to retain general image representations. This strategy was particularly effective in enhancing model sensitivity to subtle differences between bacterial and viral pneumonia, contributing to more accurate classifications.

## Experiment Setup
### Dataset
We used two datasets from Kaggle:

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

1.	Chest X-Ray Images (Pneumonia) [1]
2.	Pediatric Pneumonia Chest X-Rays [2]
### Statistics
1. Images split into training (80%), validation (10%), and test (10%).
2. Class distribution: Normal, Bacterial Pneumonia, Viral Pneumonia.
### Implementation
1. Frameworks: PyTorch and TensorFlow for model implementation.
2. Models: CNN, ResNet, DenseNet, and GAN with classifiers.
3. Hardware: Experiments were conducted on a GPU-enabled machine.
### Model Architectures
1.	CNN: Sequential convolutional layers with ReLU activations and max pooling.
2.	ResNet/DenseNet: Architectures with skip connections or dense blocks for deeper feature extraction.
3.	GAN: Generator and discriminator networks trained adversarially to augment data.

## Experiment Results
### Main Results
1.	CNN achieved over 97% accuracy in binary classification (normal vs. pneumonia), which yielded high accuracy.
2.	ResNets and DenseNets performed well in distinguishing bacterial from viral pneumonia, and both achieved approximately 75% accuracy. 
3.	GANs improved synthetic data augmentation but did not significantly enhance triple-class results. Triple classification performance plateaued at around 70% accuracy.
### Supplementary Results
1.	Parameter tuning revealed optimal learning rates and batch sizes for each model.
2.	Data augmentation techniques do improve the transfer learning model a bit.
3.	Applying a dummy class for classification job in ResNet proves meaningless.

## Discussion
### Insights
1.	Binary classification tasks benefit significantly from deep learning models.
2.	Differentiating bacterial from viral pneumonia is challenging due to subtle visual differences.
3.	The GAN model needs to adjust a wide range of parameters and it is tough to converge. Thus, further funding is strongly required.
4.	Our experiments demonstrated that our CNN model achieved results approximately 10% better than those found in existing Kaggle code implementations. This highlights the effectiveness of our optimized training process, including techniques such as gradient stabilization and tailored data augmentation.
5.	Additionally, while most publicly available implementations focus solely on binary classification (distinguishing between normal and pneumonia cases), our work goes a step further by exploring multi-class classification to differentiate between bacterial and viral pneumonia. This is an area with limited prior research or publicly available benchmarks, making our contributions particularly noteworthy in addressing this gap.

### Limitations
1.	Class imbalance, too much bacterial pneumonia data with too little viral pneumonia data, in the dataset affected triple-class classification performance.
2.	Difficulty in distinguishing bacterial from viral pneumonia due to overlapping radiological features.
3.	While our models achieved promising results in distinguishing normal cases from pneumonia, the classification of pneumonia types (bacterial vs. viral) remains a significant challenge. The current performance highlights the inherent difficulty in differentiating between these two types, likely due to the subtle radiological differences and potential dataset limitations.

### Future Work
1.	Implement attention mechanisms to focus on region-specific features in X-rays.
2.	Explore pre-training on larger datasets or additional augmentation techniques.
3.	Collaborate with radiologists to integrate clinical context into the classification pipeline.
4.	Develop more advanced modeling techniques and conduct extensive experiments to address the classification of pneumonia types (bacterial vs. viral).

## Conclusion
This project demonstrates the efficacy of deep learning models for pneumonia classification using chest X-rays. While binary tasks achieve robust performance, multi-class classification remains challenging, highlighting opportunities for further research into feature extraction and augmentation.

## References
1.	Mooney, Paul Timothy. Chest X-ray Pneumonia. Kaggle, n.d. Accessed 28 Nov. 2024. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. 
2.	Mvd, Andrew. Pediatric Pneumonia Chest X-ray. Kaggle, n.d. Accessed 28 Nov. 2024. https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray.
3.	Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua (2014). Available at: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf. Proceedings of the International Conference on Neural Information Processing Systems (NIPS 2014). pp. 2672–2680.
4.	Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional Image Synthesis with Auxiliary Classifier GANs." arXiv preprint, arXiv:1610.09585v4 [stat.ML], 20 Jul. 2017. Available at: https://arxiv.org/abs/1610.09585.
