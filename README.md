# Transfer-learning-German-and-Danish-road-sign-classification-with-CNN


# Abstract
The idea of self-driving cars commonly circulating on our streets is becoming more and more a reality as time passes. With our final project, we focused on one of the challenges that self-driving vehicles must face in order to understand the environment in which they are surrounded: the recognition of road signs.
We tested the performance of a customized Convolutional Neural Network (CNN) that was trained with a German road signs dataset, consisting of 107500 images after the data augmentation. We investigate with what result it is possible to implement a trained model with German road signs to the Danish road system. Starting from an implementation of an AlexNet architecture, we created and tested 21 models, discussing the results of the best 3 ones. We finally selected the best performing one on both German and Danish data which resulted in an accuracy and F1 of 0.91 and 0.77 respectively. After our analysis, the model is performing well on signs with a unique shape like STOP or Give Way to all Traffic but it might struggle with correctly classifying road signs with pictograms, which could include characteristics which are more complex and harder to generalize. With this paper we want also to address and discuss the idea of a unique European model which is able to adapt within different country settings and the further improvements towards this direction.

# Introduction 
In recent years, the idea of autonomous vehicles has increased in popularity, resulting in a wide discussion in the media and research field. In 2021, the United Kingdom was the first country in Europe to announce a set of regulations regarding the use of self-driving vehicles (Criddle, 2021). There is a possibility this could be a form of transport in the future. However, security is one of the most important factor to be accounted for.
One of the features of self-driving cars is the ability to detect and recognize road signs in order to circulate in areas it has not seen before. The road signs across Europe follow uniform standards: in 1968, most countries signed the Vienna Convention on Road Signs and Signals, with the objective of establishing an harmonized system across all Europe (UN, 1968). However, naturally road signs can slightly differ depending on the country. Main differences can be seen in the color coding, design of pictograms, language and text fonts (Araar et al., 2020).
Our project addresses two research questions: the first is methodological in nature as we want to test how well a machine learning model, built on a CNN architecture is able to recognize road signs within the boundaries of one country. Secondly, we want to investigate how accurately CNNs are able to generalize the features of road signs trained with a dataset from a single country, Germany and predict the signs of another 1968 Vienna Convention following country, Denmark.
The outcome of our project can be viewed as an additional source in the field of image recognition with the use of a CNN. Moreover, it can be a starting point for the self-driving cars industry in accounting for slight differences in road signs across Europe. By investigating the performance of a CNN model trained on a dataset concerning one European country to traffic signs in other countries, it can provide a possible saving solution in terms of both resources and training time and constitute the base for a unified European open- source model.

# Data description
German Traffic Sign Recognition Benchmark (GTSRB) is a dataset with 43 labels of traffic sign images.It was obtained from a public source-INI, which is the German institute for Neuroinformatic (Stallkamp et al., 2011). It contains a total of 39209 images of traffic signsbelonging to different categories, for example mandatory, warning, priority and regulatory signs.

<img width="605" alt="Screenshot 2023-01-31 at 21 43 59" src="https://user-images.githubusercontent.com/91185911/215878608-d991647a-a086-4236-89f6-588e80751dee.png">

All pictures have been captured from a car driving around with an attached camera. Therefore images have been captured from different distances and directions. The images are of varying colors, shape, resolution and quality and some images from different classes will look similar. Some examples of the different dimensions include 32x32, 27x27, 81x81 and 82x82. In figure 1, a few examples of traffic signs from different categories are shown.

The second dataset concerns Danish road signs and has been collected manually through Google Street View in the area of Copenhagen, Denmark. These images will also be taken from different directions. It includes images of 10 classes that can also be found in the German dataset. Three images per class were collected and then augmented to increase the dataset's size and variety. GTSRB will be referred to as dataset 1 and the Danish dataset with pictures from Google street view will be referred to as dataset 2.


# Framework 
In this project, we train an AlexNet model that initially does not produce satisfactory results. With the use of TensorBoard we optimize its parameters resulting in 21 configurations trained on 107 500 images.
The paper has implemented a CNN models based on the AlexNet architecture of a CNN. For the first model, the AlexNet architecture was used to train on the German dataset (Krizhevsky, Sutskever & Hinton., 2012). Subsequently this trained model was evaluated by looking at validation and training loss, also showing a poor performance when tested on the Danish traffic signs. Thereupon, it was understood that this architecture was not appropriate for our dataset and had to be tweaked accordingly. The second model is a custom CNN model based on this architecture, where the model was iterated over different combinations of layers and nodes to evaluate which one performs better on the German data. Thereafter, this model was tested on the Danish dataset of traffic signs.

# Tensor Board Optimization:
Three inner layer structures of the model will be modified and tested. First, the number of consecutive convolutional and pooling layers (1, 2, 3), second the number of neurons of each layer of (32, 64 and 128) and finally the number of dense layers (0, 1)
<img width="874" alt="Screenshot 2023-01-31 at 21 04 10" src="https://user-images.githubusercontent.com/91185911/215870265-f05bf301-1242-4613-9389-d22964cf319d.png">
Convolutional layers following the first one received twice the number of neurons, since it is common to increase the number of neurons in convolutional layers in a neural network. This is usually doubled after pooling because each pooling divides each dimension by a factor of 2. Therefore it is possible to double the amount of neurons without exploding the memory usage or computational power, as the number of features processed decreases with each pooling.
The Rectified Linear Unit (ReLu) activation function was for both the Convolutional and Dense layers with the exception of the last Dense layer, which uses the Softmax activation function for the final predictions.
Using a for loop architecture, the combinations of all model variations were tested and logged. The training of the different models took place on UCloud, allowing for machines with 64 vCPUs and 276 GB of RAM. Based on the logs, TensorBoard allows users to visually understand how the (validation) loss and (validation) accuracy rate has changed over the 10 selected epochs. By evaluating the logs via TensorBoard, the 3 best models with the highest validation accuracy were selected.

Even though the architecture with the most layers and the largest number of neurons achieved the best result, much simpler models also performed well. The model with one convolution, one dense layer and 32 units per layer achieved a validation accuracy of 96.22% and required only 10 minutes and 36 seconds of training time. In contrast, the best performing model required a training time of 1 hour and 54 minutes.


# Results
We notice very high validation accuracies for all three models. It is noticeable that architecturally the most complex model has performed the best, but above all the number of convolutional layers is decisive. The convolutional block contains a Pooling layer, a Batch Normalization and dropout in addition to the filter itself. It can be speculated that a dropout of 0.25 may be crucial at this point, as it reduces the risk of overfitting before the first Dense layer.

<img width="812" alt="Screenshot 2023-01-31 at 21 25 31" src="https://user-images.githubusercontent.com/91185911/215874663-a476e9fd-d65a-4614-8a27-9328f088fd26.png">

It is noticeable that especially for the lowest of the models, which is equipped with only one dense layer, the test accuracy has fallen sharply compared to the validation accuracy and, in parallel, the loss has risen sharply. If parameters of the only Dense layer were last adjusted by the validation step and this layer must directly predict the labels, this can have strong effects on unseen test data. In any case, it is clear that the second-placed model requires half the number of units per layer to perform almost equally on previously unseen data. From a scientific perspective, it is not the winner, but it could be preferred in the industry due to its resource efficiency and training speed.

<img width="533" alt="Screenshot 2023-01-31 at 21 29 02" src="https://user-images.githubusercontent.com/91185911/215875454-dd4a5669-f4f5-44f4-adbb-9eed939f22aa.png">

The selected models are confronted with the dataset of Danish street signs. Accordingly, a transfer performance is expected here, since the concrete designs of the street signs partly correspond to the German ones, but not completely. Therefore, it is not surprising that accuracy falls and loss rises. It is interesting here to see how the first two models differ. In the testing phase with the German data set, both still performed very close to each other. Since the difference in accuracy here is over 10%, it shows that the model with 128 neurons per layer could generalize and abstract more easily. More, smaller segments may allow to perceive changes in the whole and still to classify them correctly.

<img width="576" alt="Screenshot 2023-01-31 at 21 37 44" src="https://user-images.githubusercontent.com/91185911/215877306-e53609f9-21bf-49c1-a14d-76015008231e.png">

With accuracy split by classes with the Danish dataset, both the correct and predicted classes are included in the graph. The classes with an f1 score of zero were predicted by the model but do not exist in the Danish dataset. As expected, for classes with a high f1 score we see a great similarity between the design of the German and Danish road signs, while for low f1 scores the designs differ extensively. As an example, class 28 shows a pictogram of two people. These differ by different drawings of the persons, which explains the difficulty of generalization.

<img width="553" alt="Screenshot 2023-01-31 at 21 38 38" src="https://user-images.githubusercontent.com/91185911/215877478-8e51fefc-c994-43f2-a2d4-e132f52eedd1.png">






References:

Araar, O., Amamra, A., Abdeldaim, A., & Vitanov, I. (2020). Traffic Sign Recognition Using a Synthetic Data Training Approach. International Journal on Artificial Intelligence Tools, 29(05), 2050013.
Criddle, C.(2021). Self-driving' cars to be allowed on UK roads this year. BBC. Available from: https://www.bbc.com/news/technology-56906145
J.Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011

