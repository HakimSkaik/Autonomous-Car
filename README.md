### Autonomous-Car
Applied Deep Learning, Computer Vision and Machine Learning techniques to build a fully functional self driving car with python and Udacity open-source software

### Snippet from video of car route

https://user-images.githubusercontent.com/98859282/197321927-8ef3ca48-64bc-4f07-a9eb-be102394034a.mp4

### Open-source libraries used 

* <a href="https://pypi.org/project/keras/" target="_blank">Keras</a>
* <a href="https://www.tensorflow.org/install/pip" target="_blank">TensorFlow</a>
* <a href="https://pypi.org/project/opencv-python/" target="_blank">OpenCv</a>
* <a href="https://numpy.org/install/" target="_blank">NumPy</a>
* <a href="https://matplotlib.org/stable/users/installing/index.html" target="_blank">Matplotlib</a>
* <a href="https://pandas.pydata.org/docs/getting_started/install.html" target="_blank">Pandas</a>
* <a href="https://pypi.org/project/imgaug/" target="_blank">imgaug</a>
* <a href="https://pypi.org/project/path.py/" target="_blank">ntpath</a>
* <a href="https://github.com/udacity/self-driving-car-sim" target="_blank">Udacity Self Driving Car Simulator</a>




## *Identifying Lanes* 

Identified lines inside of a gradient image with hough transfor technique and placed these lines on a black image which has the same dimensions as the road image, thereby, by blending the two, I was able to place the detected lines nack into the original "test_image.jpg"

![image](https://user-images.githubusercontent.com/98859282/197322576-635ad237-f533-43a5-b908-1c62ff4ed126.png)  ![image](https://user-images.githubusercontent.com/98859282/197322583-f81e59a1-bbec-4755-b279-77f166c4d10b.png) ![image](https://user-images.githubusercontent.com/98859282/197323178-bb34747d-4f6f-4d18-bc7a-cee605afad20.png) ![image](https://user-images.githubusercontent.com/98859282/197323286-135179e0-b2be-4b6e-b654-3b3f1a6c47ca.png)


## *Neural Network Training* 

Successfully trained a neural netwok to learn how to classify training data, reaching 97% accuracy

![image](https://user-images.githubusercontent.com/98859282/197323690-01472f4b-08af-412e-a9b6-7f26dfa64684.png)

**In "Deep-neural-network.ipynb" file:**

I utilized linear models as a building block in the neural network to obtain non-linear models which best classifies the data.

Made use of a gradient descent optimization algoritim which acts to minimize the error of model by iteratively moving in the dierction with the steepest decent, the direction which updates the paramters of models while ensuring minimal error and updates the weight of every model and every single layer in order to attain the final model

Coded a deep neural network that's properly able to calssify previously labeled data to then make predictions on newly inoutted data; at about 48 epochs, I otained a model that reaches 99% accuracy.



![image](https://user-images.githubusercontent.com/98859282/197324543-5d6f7b79-069e-4b79-836a-35030229de9a.png)


## *Traffic Signs Prediction*

in the "MNIST deep learning.ipynb":

I successfully trained a neural network to have a model fit image data,to make predictions between 42 different classes of traffic signs, where it's able to recognize the traffic class based on the inputted image.



![image](https://user-images.githubusercontent.com/98859282/197324717-1719b923-308d-46ff-a6c9-436d7c2e9f9e.png)  ![image](https://user-images.githubusercontent.com/98859282/197324800-0b1b9c02-21f8-4518-9fcd-891af919387c.png)  ![image](https://user-images.githubusercontent.com/98859282/197324828-80b7bca8-dcd5-4f8e-923e-bbce28f60ba6.png)



## *Convolutional Neural Networks*


Detecting meaningful features when given various of different images and labels is no easy task, which is convolutional layers are key players in a convolutional network.

**Let's compare Convolutional Networks VS. Deep Neural Networks**


![image](https://user-images.githubusercontent.com/98859282/197325163-56e340fc-32d0-40fd-8d20-3e70acdc242a.png)


First and foremost, we see that our convolutional network is performing better than the previous deep network used to classify the data. As seen, the accuracy is much higher at ninety nine or rather than 92 and 94, which is a big plus.

Also, notice how small the losses are compared, the errors are at 0.05 and lower,
whereas on the deep neural network it's above 0.2.

And third, the difference in accuracy between the training and validation set is relatively low, which
means that the network is performing with a similar degree of accuracy on the training data and the
validation data.

This implies that our solution for using convolutional networks to classify these images was successful.


## *Dropout Regularzation for Neural Networks*

Dropout is a regularization technique for neural network models proposed by Srivastava et al. in their 2014 paper “Dropout: A Simple Way to Prevent Neural Networks from Overfitting” <a href="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf" target="_blank">pdf</a>.

Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass, and any weight updates are not applied to the neuron on the backward pass.

As a neural network learns, neuron weights settle into their context within the network. Weights of neurons are tuned for specific features, providing some specialization. Neighboring neurons come to rely on this specialization, which, if taken too far, can result in a fragile model too specialized for the training data. This reliance on context for a neuron during training is referred to as complex co-adaptations.

You can imagine that if neurons are randomly dropped out of the network during training, other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.

The effect is that the network becomes less sensitive to the specific weights of neurons. This, in turn, results in a network capable of better generalization and less likely to overfit the training data.

As noticed from the results of trained model, the validation accuracy jumps up to match the training accuracy 


![image](https://user-images.githubusercontent.com/98859282/197325721-ae403246-25db-401e-94e6-34db70655efb.png)


It was due to the convolutional network that clearly resulted in improved accuracy and less overfitting, which means that the network was trained effectively.
I even visualized the outputs of these convolutional layers to get a solid understanding of just how these convolutional layers behave.


## *Behaviuoral Cloning* 

the "Behavioural_Cloning Final.ipynb" consists of convolutional neural networks for which I took images at each instance of the drive, which were then used to represent the training dataset, the label for each image is the steering angle of the car at that specific instance. I displayed all of these images to the convulutional neural network and allowed it to learn how to drive autonomously, as the model will learn to adjust the steering angle to an appropriate degree bsaed on the situation that it finds itself in.
 
 
 ### <a href="https://developer.nvidia.com/blog/deep-learning-self-driving-cars/" target="_blank">NVIDIA Neural Model</a>
 
 I utilized YUV color space for the dataset as opposed to the default RGB format or a greyscale image, and this is because experts say that this color is very effective for use in training as it adds colors to the image. I also applied the gaussian blur since it reduces moise within the image 
 
 ![image](https://user-images.githubusercontent.com/98859282/197326339-71aa662d-d863-47bf-99c9-66e01dcccb27.png)
 
 
## *Socket.io*
 
Sockets in general are used to perform real time communication between a client and a server when a client creates a single connection to a web socket server. It keeps listening for new events from the server allowing us to continuously update the client with data.
 
The goal was to create a bi directional client server communication but ultimately as a result create a connection between the model which was loaded into atom, the simulator and server would then be initialized with socket.io so I implemented a fully compliant socket IO web server and now having specified
the server I required a middleware, to dispatch traffic to a socket io web application.

## *WSGI Server*
 
To sum, I combined the socket server with a flask webapp.  I used the WSGI Server for calling convention for web servers to forward requests to web applications or frameworks written in Python, these requests made my by the client are sent to the web application itself to launch this WSGI server.

 
 

