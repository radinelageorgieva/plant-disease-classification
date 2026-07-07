# Plant disease classification using Transfer learning
A PyTorch-based computer vision project utilizing EfficientNet-B0 transfer learning to classify plant diseases, featuring imbalance handling and explainable AI using Grad-CAM.



## Project architecture

1. **MLflow Integration**: MLOps tracking setup to log training metrics, parameters, and loss/accuracy curves in real time.
2. **Data preprocessing and augmentation**: Image resizing to 224x224 pixels, random horizontal/vertical flips, color jitter, and standard ImageNet channel normalization.
3. **Exploratory Data Analysis (EDA)**: Class distribution maps to analyze data quality and sample distributions across all categories.
4. **Dataset Stratification**: Stratified splitting strategy using 90% for training and 10% for validation to ensure equal class ratios in both splits.
5. **Class imbalance mitigation**: Implementation of a WeightedRandomSampler to prevent model bias toward majority classes.
6. **Transfer learning setup**: Freezing the pre-trained EfficientNet-B0 backbone and replacing the classifier head with a custom Sequential layer (`Dropout(p=0.2)` + `Linear`).
7. **Two-phase optimization**: Phase 1 feature extraction using the Adam optimizer ($lr=10^{-4}$), followed by phase 2 deep fine-tuning of Block 8 using the AdamW optimizer ($lr_{block8}=10^{-5}$, $lr_{classifier}=10^{-4}$).
8. **Model testing**: Evaluate the finalized weights of the trained model against the independent test dataset to obtain the definitive baseline accuracy of the computer vision workflow.
9. **Evaluation and visualization**: Learning curves: Generated dynamically using the MLflowClient API to plot Loss vs. Epochs and Accuracy vs. Epochs; Confusion matrix ; F1 score
10. **Failure modes analysis**:  Model confidence analysis on misclassified images ;  Quantitative error pair ranking
11. **Grad-CAM**: Apply Gradient-weighted Class Activation Mapping (Grad-CAM) to validate the visual reliability of the network. This technique generates colorful heatmaps over the leaves to show exactly which pixel areas and lesions the model focuses on to make its final decision.

---

## Libraries

* PyTorchTorchvision
* MLflow
* OpenCV (cv2)
* PIL
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Tqdm
* Copy
* Os
* Random


## Model diagnostics & Interpretability

### 1. Visual inspection framework (True vs. Predicted)
I implemented a random sampling workflow that extracts and visualizes subsets of correct and incorrect predictions from the test set. Correct classifications are mapped onto a green grid, while incorrect choices are mapped onto a red grid. This step allows for an immediate, qualitative health check of the model’s prediction boundaries on random leaf samples.

### 2. Prediction confidence analysis on misclassified images
I calculated the network's raw Softmax probabilities to categorize the model's errors into two distinct failure profiles:
* **High-confidence mistakes (80% to 95% confidence)**: Instances where the model is completely certain of its wrong answer (e.g specific corn diseases). This signals severe systematic visual bottlenecks and heavy texture overlaps in the underlying dataset.
* **Low-confidence errors (under 35% confidence)**: Instances where the model's confidence drops significantly and the probability gap between the top classes is minimal. This indicates that the network is simply guessing between two visually similar-looking leaf spots (e.g., fine-grained Tomato leaf spots).

### 3. Visualizing failure modes with Grad-CAM
To open the neural network's *black box* and inspect these critical boundaries, I integrated a custom **Grad-CAM (Gradient-weighted Class Activation Mapping)** workflow. By capturing the gradients flowing into the final convolutional feature layer and upscaling the raw activations, the script generates 2D color heatmaps overlaid onto the original leaf shapes. 


---

## Conclusion


This is a deep learning project for plant disease classification using transfer learning with the EfficientNet-B0 model. The final model achieved a 91% overall test accuracy. А 91% accuracy represents a good and reliable performance given the difficulty of a 38 classes problem, which spans across 12 different plant species and includes 26 disease categories and 12 healthy leaf categories. \
To reach this performance, the project combined several data engineering and model training methods.\
The dataset contains two main folders: train and test. To create a reliable validation set, I extracted exactly 10% of the images from every single class folder inside the train dataset. I then shuffled these extracted images to build a stratified validation set. This strategy ensures that the training set (90%) and the validation set (10%) have the exact same class distribution and that every disease is equally represented.\
To handle the high class imbalance shown in the EDA, I implemented a balancing strategy using a WeightedRandomSampler. This calculated a specific weight for each disease category based on the training samples, successfully preventing the network from developing a bias toward majority classes and allowing the model to achieve high overall accuracy across both rare and common diseases.\
The model architecture was optimized using a precise two-phase transfer learning approach. In phase 1 I utilized feature extraction by freezing all the deep convolutional layers of the pre-trained EfficientNet-B0 base model. Then I replace the default classifier head with a custom sequential architecture containing a Dropout layer with a probability of 0.2 for regularization, a Linear layer to map the 1,280 extracted features to our 38 output classes. This phase was trained using the Adam optimizer with a learning rate of 0.0001 and Cross-Entropy Loss, achieving a baseline validation accuracy plateau of 87.79% by epoch 6. To break past this performance limit I continue to phase 2 - fine-tuning stategy by unfreezing the final convolutional block 8 of EfficientNet-B0 architecture. I trained the model for 5 additional epochs using the AdamW optimizer with discriminative learning rates, applying a very small learning rate of 0.00001 for Block 8 to gently adapt the deep visual filters without causing catastrophic forgetting, and a learning rate of 0.0001 for the classifier head. This fine-tuning strategy broke the plateau and pushed the final overall test accuracy to 91%.\
The true engineering value of this project is in deep statistical and visual investigation of the model's failure modes. By computing a quantitative error pair ranking in the most frequent misclassification pairs, I shifted the focus from individual images to systematic bottlenecks. The normalized confusion matrix showed strong diagonal strength with some missclasification. Some classes have perfect 1.00 classification scores like Orange Haunglongbing and Corn healthy. However, the quantitative ranking exposed a critical systematic limitation in fine-grained visual differentiation, showing that the model recorded 195 misclassifications between Corn Northern Leaf Blight and Corn Cercospora Leaf Spot, followed by 109 errors between Tomato Septoria and Early Blight.\
The confidence analysis revealed that these mistakes fall into two distinct failure categories. First, high-confidence misclassifications, where the model predicts the wrong class with 80% to 95% confidence because the long brown spots on different corn leaves have a massive visual overlap in dataset textures, completely misleading the network. Second low-confidence errors, where the model's confidence drops below 35% and the probability gap between the true and predicted class is minimal, meaning the model is simply guessing between two highly similar-looking leaf spots.
To open the "black box" of the deep neural network and validate its actual reliability, I integrated a custom Grad-CAM workflow. By tracking the class gradients flowing into the final convolutional feature layer, this program generated visual 2D spatial activation heatmaps overlaid onto the original leaf images specifically for the misclassified images (and only 1 true predicted).  The analysis of these heatmaps provided the most critical insight of the project by exposing how the model gets tricked when it fails. The images proved that for incorrect predictions, the network completely ignores the actual disease spots on the leaf surface. Instead, the program's red activation zones showed that the model relies on geometric shortcuts, locking onto sharp leaf tips, outlines, or gets completely distracted by environmental noise as background shadows. Тhis analysis highlights that while the 91% test accuracy is built on genuine feature recognition, transitioning this program into a real-world agricultural tool can be significantly optimized by incorporating automated background removal to keep the convolutional layers strictly focused on the leaf surface.\
In summary while a 91% accuracy indicactes that transfer learning with EfficientNet-B0 can be effective for large-scale agricultural classification, but our deep error analysis shows that the model is highly sensitive to background noise and leaf geometry. This project demonstrates that achieving high accuracy on a dataset does not mean the model is looking at the correct biological patterns. To transition this system from a prototype to a real-world, production-ready agricultural deployment, future work must incorporate automated background removal, image segmentation, or spatial attention masking to force the convolutional layers to look strictly at the inner leaf pathology.

Future work and ideas for improving accuracy
Since a 91% accuracy shows that the model looks at the leaf surface but still struggles to evaluate fine-grained details, future improvements should focus on the following strategies:
* Ensemble learning: combine the predictions of EfficientNet-B0 with other architectures, such as Vision Transformers or ResNet. Using a voting system between multiple models can balance out individual mistakes and can boost accuracy.
* Multi-task learning: train the network to output two predictions at the same time - the plant species like corn or tomato and the specific disease. This forces the model to understand the plant type first, narrowing down the choices and helping it evaluate the leaf surface much better.
* Advanced data augmentation: implement cutting-edge augmentation techniques like MixUp or CutMix, which blend pieces of different diseased leaves into a single image. This forces the network to learn strict pixel-level details instead of memorizing the general look of a leaf.
* Lesion-focused Object Detection: integrate a lightweight object detection model (like YOLO) before the main classifier. This setup will first locate and crop only the actual disease spots on the leaf, feeding the model with close-up textures of the disease and removing all surrounding distractions.


## References

* **Plant Diseases Dataset**: Saroz, O. Plant Diseases Dataset on Kaggle. Dataset repository: [saroz014/plant-diseases](https://kaggle.com).
* **EfficientNet Architecture**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *International Conference on Machine Learning (ICML)*, arXiv:1905.11946. Document: [arxiv.org/pdf/1905.11946](https://arxiv.org).
* **MLflow Platform**: MLflow AI Platform Team. (2018). Accelerating the Machine Learning Lifecycle with MLflow. Python API Reference Documentation: [[mlflow.org/docs](https://mlflow.org).](https://mlflow.org/docs/latest/ml/projects/)
* **PyTorch Library**: PyTorch Development Team. Training a CNN Classifier (Deep Learning with PyTorch: A 60 Minute Blitz). Documentation: [[pytorch.org/tutorials](https://pytorch.org).](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* **Vision Inspection CNN**: Towards Data Science. Building a Vision Inspection CNN for an Industrial Application. Practical guide:[ [towardsdatascience.com].](https://towardsdatascience.com/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a/)
* * **Grad-CAM in PyTorch**: Simonelli, J. (2023). Model Explainability with Grad-CAM in PyTorch. *Julius' Data Science Blog*. Technical tutorial: [jss367.github.io](https://jss367.github.io/model-explainability-with-grad-cam-in-pytorch.html).
* * Vision Inspection CNN: Nowitzky, I. 2024. Building a Vision Inspection CNN for an Industrial Application. Towards Data Science. Practical guide: https://towardsdatascience.com/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a/
* * EfficientNet Transfer Learning: Rath, S. R. 2022. Transfer Learning using EfficientNet PyTorch. DebuggerCafe. Technical tutorial: https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

