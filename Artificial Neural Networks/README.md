#Artificial Neural Networks (ANN)

<img height="450" src="images\giffy.gif" width="700"/>

**Have you ever wondered how our brain works?** We all might have learned about it in our school days, Right! Our nervous system helps our brain to send all the messages to the desired parts of our body. **ANN** works exactly similar to how the neurons work in our nervous system.   

##Introduction to Artificial Neural Networks

- **"Artificial Neural Networks"** are the most popular ***Machine Learning*** Algorithms today.
- The invention of these Neural Networks took place in the **1970's**, coined by **Warren S McCulloch** and **Walter Pitts** but, they have achieved the huge popularity due to recent increase in computation power because of which they are now virtually everywhere.
- In every application that we are use today, there is, ***Neural Networks*** power, the intelligent interface, that keeps us engaged. 
- The term **"Artificial Neural Network"** refers to a biologically inspired sub-field of **Artificial Intelligence** modeled after the brain.

##What is Artificial Neural Network (ANN)?
- **ANN (Artificial Neural Network)** is at the core of the very core of **Deep Learning** an **advanced version** of **Machine Learning techniques**.
- **ANN** are special type of machine learning algorithms that are modeled after the human brain.
- Similarly, as the human brain has neurons interconnected with each other, artificial neural networks also have neurons that are linked to each other in various layers of the network. These neurons are known as **nodes**.
- That is, just like how the neurons in our nervous system are able to learn from the past data, similarly, the **ANN** is able to learn from the data and provides us the responses in the form of predictions or classification.


  <center>
    <img height="350" src="images\neuron.jpg" width="500"/>

**Biological Neural Network**

<br><br>
<img height="250" src="images\artificial-neural-network3.png" width="550"/>

**Artificial Neural Network**
</center>
<br>

**" Neural Networks** _is a computational learning system that uses_a network of functions to understand and translate a data input of one form into a desired output, usually in another form. The concept of the **Artificial Neural Network** was inspired by human biology and the way **neurons** of the human brain function together to understand inputs from the human senses._ **"**
<br>
- ANN's are non-linear statistical model which displays a complex relationships between the inputs and the outputs to discover a new pattern.
- Variety of tasks such as, _**Image Recognition**, **Machine Translation**, **Medical Diagnosis**_, makes use of **Artificial Neural Networks**.

###Relationship Between _Biological Neural Network_ and _Artificial Neural Networks_ :

| Biological Neural Network | Artificial Neural Network |
| :----: | :----: |
| Dendrites | Inputs |
| Cell Nucleus | Nodes |
| Synapses | Weights |
| Axon | Output |


##Architecture of Artificial Neural Networks
- To understand the concept of the architecture of an **Artificial Neural Networks**, we need to understand, **_What a neural network consists of ?_** 
- In order to define a neural network that consists of a large number of artificial neurons, which are termed units arranged n a sequence of layers.
- Let us look at the various types of layers available in neural networks.
- Artificial Neural Network primarily consists of three layers :

<center>
    <img height="300" src="images\artificial-neural-network4.png" width="600"/>

**Artificial Neural Network Architecture**
</center>

###1. Input Layer :
- **Input Layer**, as the name suggests, it accepts input information in several formats like : _texts, numbers, audio files, image pixels, etc.,_ provided by the programmer.
- Along with the input variables, it also represents the bias term. Hence, if there are n input variables, the size of the input layer is **n + 1**, where, **+ 1** is the bias term. 
 
###2. Hidden Layer/ Layers :
- **Hidden Layer/s** is present between input and output layer.
- It is a set of neurons that performs all the calculations to find hidden features and patterns.

###3. Output Layer :
- The **Output Layer** is the conclusions of the model derived from all the computations performed.
- There can be single or multiple nodes in _output layer_.
- Like, if we have the binary classification problem, the output node would be 1 but, in the case of multi-class classification, the output node would be more than one.


- In a neural network, there are multiple parameters and hyper-parameters that affects the performance of the model.
- The output of the ANN's model is mostly dependent on these parameters.
- Some of these parameters are:
  - **weights**,
  - **biases**,
  - **learning rate**,
  - **batch size**, etc,.
- Each node in ANN has same weight assigned to it.
- A transfer function is used for calculating the weighted sum of the inputs and the bias.

<center>
<img height="260" src="images\how-ANN-works.png" width="550"/>

</center>

### **_What is an activation function?_** and **_Why do we use them?_**
- After the transfer function has calculated the sum, the **_activation function_** obtains the result.
- Based on the output received, the activation functions fire the appropriate result from the node.
- **For Example**, if the output received is above 0.5, the activation function fires a 1 otherwise, it would remain 0.
- The purpose of the activation function is to **_introduce the non-linearity_** into the output of a neuron.
- Some popular activation functions used are:
  - Sigmoid 
  - RELU  
  - Softmax 
  - tanh
  

  <center>
  <img height="250" src="images\How-artificial-Neural-Networks-work.png" width="600"/>
  </center>
  
####Explanation
- We know, the neural network has neurons that works in correspondence of weight, bias and their respective activation functions.
- In neural network, we would update the weight and biases of the neurons on the basis of the error at the output.
- This process is known as **_back propagation_**.
- ####What is Back Propagation actually?
  - _**" Back Propagation is the process of updating and finding the optimal values of weights or coefficients which helps the model to minimize the error i.e difference between the actual and predicted values."**_
- Activation makes the back propagation possible since the gradients are supplied along with the error to update the weight and biases.

####Back Propagation with Gradient Decent 
- **Gradient Descent** is one of the **optimizers** which helps in calculating the new weights. 
- Let’s understand step by step how Gradient Descent optimizes the cost function.
- In the image below, the curve is our cost function curve and our aim is to minimize the error such that Jmin i.e global minima is achieved.

<center>
<img height="300" src="images\gradient.png" width="500"/>
</center>

####Steps to achieve global minima
1. First, the weights are initialized randomly i.e random value of the weight, and intercepts are assigned to the model while forward propagation and the errors are calculated after all the computation.
2. Then the gradient is calculated i.e derivative of error with respect to current weights.
3. Then new weights are calculated using the below formula, where **a** is the **learning rate** which is the parameter also known as step size to control the speed or steps of the backpropagation. It gives additional control on how fast we want to move on the curve to reach global minima.

<center>
<img height="300" src="images\update_formula.png" width="500"/>
</center>

4. This process of calculating the new weights, then errors from the new weights, and then updation of weights continues till we reach global minima and loss is minimized. 

- A point to note here is that the learning rate i.e **a** in our weight updation equation should be chosen wisely. 
- Learning rate is the amount of change or step size taken towards reaching global minima. 
- It should not be very small as it will take time to converge as well as it should not be very large that it doesn’t reach global minima at all. 
- Therefore, the **learning rate** is the **hyperparameter** that we have to choose based on the model.

<center>
<img height="300" src="images\curve.png" width="500"/>
</center>

####Why do we need Non-linear activation functions ?
- A neural network without an activation function is essentially just a linear regression model.
- The activation function does the non-linear transformation to the input, making it capable to learn and perform more complex tasks.

####Mathematical Proof

- Suppose we have a neural network like;
<center>
<img height="300" src="images\neuralNet.png" width="500"/>
</center>

####Elements of the diagram :-
  - **Hidden Layer i.e. layer 1 :-**

```markdown
z(1) = W(1)X + b(1)
a(1) = z(1)
Here,

- z(1) is the vectorized output of layer 1
- W(1) be the vectorized weights assigned to neurons of hidden layer 
  i.e. w1, w2, w3 and w4
- X be the vectorized input features i.e. i1 and i2 
- b is the vectorized bias assigned to neurons in hidden layer i.e. b1 and b2
- a(1) is the vectorized form of any linear function.

(Note: We are not considering activation function here)
```

 - **Layer 2 i.e. output layer :-**

```markdown
//  Note : Input for layer 
//   2 is output from layer 1
z(2) = W(2)a(1) + b(2)  
a(2) = z(2) 
```

- **Calculations at output layer :-**

```markdown
// Putting value of z(1) here

z(2) = (W(2) * [W(1)X + b(1)]) + b(2) 

z(2) = [W(2) * W(1)] * X + [W(2)*b(1) + b(2)]

Let, 
    [W(2) * W(1)] = W
    [W(2)*b(1) + b(2)] = b

Final output : z(2) = W*X + b
which is again a linear function.
```

This observation results again in a linear function even after applying a hidden layer.

Hence, we can conclude that, *doesn't matter how many hidden layer we attach in neural network, all layers will behave same way because **the composition to two linear function is linear function itself***.

Neuron cannot learn with just a linear function attached to it. 

A non-linear activation function will let it learn as per the difference with respect to the error.

**Hence, we need an activation function.**

###Variants of Activation Functions
####1. Linear Function
- **Equation :** 
  - Linear function has the equation similar to as of a straight line i.e. y = ax + b.
  - No matter how many layers we have, if all are linear in nature, the final activation function of last layer is nothing but just a linear function of the input of first layer.
- **Range :** -inf to +inf
- **Uses :** Linear activation function is just used at one place i.e. *output layer*.
- **Issues :**  
  - If we will differentiate linear function to bring non-linearity, result will no more depend on input “x” and function will become constant, it won’t introduce any ground-breaking behavior to our algorithm.
- **For example :** 
  - Calculation of price of a house is a regression problem. 
  - House price may have any big/small value, so we can apply linear activation at output layer. 
  - Even in this case neural net must have any non-linear function at hidden layers.
####2. Sigmoid Function
- It is a function which is plotted as ‘S’ shaped graph.
- **Equation :**
  - A = 1/(1 + e-x)
- **Nature :**
  - Non-linear. Notice that X values lies between -2 to 2, Y values are very steep. 
  - This means, small changes in x would also bring about large changes in the value of Y.
- **Value Range :** 0 to 1
- **Uses :**
  - Usually used in output layer of a binary classification, where result is either **0** or **1**, as value for sigmoid function lies between **0** and **1** only so, result can be predicted easily to be **1** if value is greater than **0.5** and **0** otherwise.
####3. Tanh Function
- The activation that works almost always better than sigmoid function is Tanh function also knows as **Tangent Hyperbolic function**.
- It’s actually mathematically shifted version of the sigmoid function. 
- Both are similar and can be derived from each other.
- **Equation :**
```markdown
f(x) = tanh(x) = 2/(1 + e-2x) - 1
OR
tanh(x) = 2 * sigmoid(2x) - 1
```
- **Range :** -1 to +1
- **Nature :** Non-linear
- **Uses :**
  - Usually used in hidden layers of a neural network as it’s values lies between -1 to 1 hence the mean for the hidden layer comes out be 0 or very close to it, hence helps in centering the data by bringing mean close to 0. 
  - This makes learning for the next layer much easier.
####4. ReLU Function
- Stands for **Rectified Linear Unit**. 
- It is the most widely used activation function. 
- Chiefly implemented in **_hidden layers_** of Neural network.
- **Equation :**
  - A(x) = max(0,x). 
  - It gives an output x if x is positive and 0 otherwise.
- **Range :** [0, inf)
- **Nature** :
  - Non-linear, which means we can easily backpropagate the errors and have multiple layers of neurons being activated by the ReLU function.
- **Uses :**
  - ReLu is **_less computationally expensive_** than tanh and sigmoid because it involves _**simpler mathematical operations**_. 
  - At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.

- In simple words, _RELU **learns much faster** than sigmoid and Tanh function_.
####5. Softmax Function
- The softmax function is also a type of sigmoid function but is handy when we are trying to handle classification problems.
- **Nature :** Non-linear
- **Uses :** 
  - Usually used when trying to **_handle multiple classes_**. 
  - The softmax function would **_squeeze the outputs_** for each class between **0** and **1** and would also **_divide by the sum of the outputs_**.
- **Output :**
  - The softmax function is ideally **_used in the output layer_** of the classifier where we are actually trying to attain the probabilities to define the class of each input.

###Choosing the correct Activation Function 
- The basic **_Thumb Rule_** is,  _" if you really **don’t know what activation function to use**, then **_simply use ReLU_** as it is a general activation function and is used in most cases these days "._ 
- **_If_** your output is for **_binary classification_** then, **_sigmoid function_** is very natural choice **_for output layer_**.

##Types of Artificial Neural Network
- There are two important types of artificial neural networks.
  1. FeedForward Neural Network
  2. FeedBack Neural Network
  
####1.  FeedForward Neural Network :
- In the **feedforward** ANNs, the flow of information takes place only in **one direction**. 
- That is, the flow of information is from the input layer to the hidden layer and finally to the output. 
- There are no feedback loops present in this neural network. 
- These type of neural networks are mostly used in **_supervised learning_** for instances such as classification, image recognition etc. 
- We use them in cases where the data is not sequential in nature.

<center>
<img height="300" src="images\feed_forward_ann.jpg" width="550"/>
<br><br>
<img height="300" src="images\feedforward-artificial-neural-networks.png" width="500"/>
</center>

####2. FeedBack Neural Network :
- In this type of ANN, the output returns into the network to accomplish the best-evolved results internally. 
- As per the University of Massachusetts, Lowell Centre for Atmospheric Research, the feedback networks feed information back into itself and are well suited to solve optimization issues. The Internal system error corrections utilize feedback ANNs.

<center>
<img height="400" src="images\feedback_ann.jpg" width="300"/>

</center>

##Bayesian Networks
- These type of neural networks have a probabilistic graphical model that makes use of Bayesian Inference for computing the probability. 
- These type of Bayesian Networks are also known as **Belief Networks** or **Bayes Networks** or **Bayes Nets**. 
- In these Bayesian Networks, there are edges that connect the nodes representing the probabilistic dependencies present among these type of random variables. 
- The direction of effect is such that if one node is affecting the other then, they fall in the same line of effect. 
- Probability associated with each node quantifies the strength of the relationship. Based on the relationship, one is able to infer from the random variables in the graph with the help of various factors. 
- The only constraint that these networks have to follow is it cannot return to the node through the directed arcs. 
- Therefore, Bayesian Networks are referred to as **Directed Acyclic Graphs (DAGs)**. 
- These Bayesian Networks can **handle** the **multivalued variables** and they comprise of two dimensions.
  1. Range of Prepositions
  2. Probability that each preposition has been assigned with.
  

- Assume that there is a finite set of random variables such that each variable of the finite set is denoted by X = {x1, x2… xn} where each variable X takes from the values present in the finite set such that Value{x1}. 
- If there is a directed link from the variable Xi to the variable Xj, then Xi will be the parent of Xj that shows the direct dependencies between these variables.
- With the help of Bayesian Networks, one can combine the prior knowledge as well as the observed data. 
- Bayesian Networks are mainly for learning the causal relationships and also understanding the domain knowledge to predict the future event. 
- This takes place even in the case of missing data.

##Why do we need ANN?
- **Utilizing the Tensorflow**, Neural Network Playground will provide a better understanding of the advantages of Neural Networks over ensemble techniques, logistic regression, or Support Vector Machine.


<center>
<img height="300" src="images\1.png" width="300"/>
<img height="300" src="images\2.png" width="300"/>
<img height="300" src="images\3.png" width="300"/>

####Classification of data points to their respective colors using the flower dataset.
</center>

- In trying to classify different coordinates (say x and y) when displayed in a two-dimensional space represents the image of a flower into their respective colors. 
- Logistic regression & neural networks were utilized for classifying coordinates into specific colors. 
- Logistic regression works with decision boundaries, hence to understand the impact of logistic regression in identifying the colors of the dots, it’s important to identify the decision boundary. 
- The output of the logistic regression clearly shows that the algorithm uses a linear decision boundary to classify the dots to their respective colors, hence ends up misclassifying a lot of data points. 
- A neural network with one hidden layer produced an improved decision boundary resulting in higher accuracy.


<center>
<img height="900" src="images\4.png" width="600"/>

####Illustrates the impact of multiple neurons on the final output. The higher the number of neurons, the better is the classification accuracy. 
</center>

##Practical Implementation with Artificial Neural Network
- ###Churn Modelling Problem
  - This a problem-solving data analytics challenge for a bank.
  - Here, we have a dataset, with a large sample of the bank's customers.
  - To make this dataset, the bank gathered information such as customer id, credit score, gender, age, tenure, balance, if the customer is active, has a credit card, etc. 
  - During 6 months, the bank observed if these customers left or stayed in the bank.
  <br><br>
  - **Goal :** 
    - To make an Artificial Neural Network that can predict, based on geo-demographical and transactional information given above, if any individual customer will leave the bank or stay (customer churn). 
    - Besides that, we have to rank all the customers of the bank, based on their probability of leaving. 
    - To do that, we need to use the right Deep Learning model, one that is based on a probabilistic approach.
    <br><br>
    
  - **Dataset :**
    - https://github.com/AmirAli5/Deep-Learning/blob/d1f9e69f2d144744434068a619b39a2d9afa039a/1.Supervised%20Deep%20Learning/1.Artificial%20Neural%20Network/Dataset/Churn_Modelling.csv

###1. Data Preprocessing
####1.1 _Import the libraries_
- In this step, we import three Libraries in Data Preprocessing part. 
- A library is a tool that you can use to make a specific job. 
- First, we import the **numpy library** used for _multidimensional array_ then import the **pandas library** used to _import the dataset_ and in last we import **matplotlib library** used for _plotting the graph._

<center>
<img height="150" src="images\1.1.png" width="300"/>
</center>

####1.2 _Import the Dataset_
- In this step, we import the dataset to do that we use the pandas library. 
- After importing our dataset we define our dependent and independent variable. 
- Our independent variables are 1 to 12 attributes as you can see in the sample dataset which we call ‘X’ and dependent is our last attribute which we call ‘y’ here.

<center>
<img height="100" src="images\1.2.png" width="450"/>
</center>

####1.3 _Encoding the Categorical Data_
- In this step, we Encode our categorical data. 
- If we see our dataset then Geography & Gender attribute is in Text and we Encode these two attributes in this part use the LabelEncoder and OneHOTEncoder from the Sklearn. Processing library.

<center>
<img height="200" src="images\1.3.png" width="600"/>
</center>

####1.4 _Split the dataset for test and train_
- In this step, we split our dataset into a test set and train set and an 80% dataset split for training and the remaining 20% for tests. 
- Our dataset contains 10000 instances so **8000 data** we **train** and **2000 data** for the **test**.

<center>
<img height="100" src="images\1.4.png" width="800"/>
</center>

####1.5 _Feature Scaling_
- Feature Scaling is the most important part of data preprocessing. 
- If we see our dataset then some attribute contains information in Numeric value some value very high and some are very low if we see the age and estimated salary. 
- This will cause some issues in our machinery model to solve that problem we set all values on the same scale there are two methods to solve that problem first one is Normalize and Second is Standard Scaler.

<center>
<img height="250" src="images\1.5.png" width="550"/>
</center>

- Here we use standard Scaler import from Sklearn Library.

<center>
<img height="150" src="images\1.5_2.png" width="450"/>
</center>

###2. Building our Model
In this part, we model our Artificial Neural Network model.

####2.1 _Import the Libraries_
- In this step, we import the Library which will build our ANN model. 
- We import **Keras Library** which will build a deep neural network based on Tensorflow because we use Tensorflow backhand. 
- Here we import the two modules from Keras. The first one is **Sequential** used for initializing our ANN model and the second is **Dense** used for adding different layers of ANN.

<center>
<img height="120" src="images\2.1.png" width="550"/>
</center>

####2.2 _Initialize our ANN model_
- In this step, we initialize our Artificial Neural Network model to do that we use sequential modules.

<center>
<img height="80" src="images\2.2.png" width="400"/>
</center>

####2.3 _ Adding the input layer and first hidden layer_
- In this step, we use the Dense model to add a different layer. 
- The parameter which we pass here first is **output_dim=6** which defines _hidden layer=6_, the second parameter is **init= uniform** basically this is a uniform function that randomly initializes the weights which are close to 0 but not 0. 
- The third parameter is **activation= relu** here in the _first hidden layer_ we use _relu activation_. 
- And the last parameter which we pass in dense function is **input_dim= 11** which means the input node of our Neural Network is 11 because our dataset has 11 attributes that’s why we choose 11 input nodes.

<center>
<img height="60" src="images\2.3.png" width="700"/>
</center>

####2.4 _Adding the Second Hidden layer_
- In this step, we add another hidden layer.

<center>
<img height="60" src="images\2.4.png" width="600"/>
</center>

####2.5 _Adding the output layer_
- In this step, we add an output layer in our ANN structure **output_dim= 1** which means one output node.
- Here, we use the **sigmoid function** because our target attribute has a binary class which is one or zero that’s why we use sigmoid activation function.

####2.6 _Compiling the ANN_
- In this step, we compile the ANN to do that we use the compile method and add several parameters the first parameter is **optimizer = Adam** here use the optimal number of weights. 
- So for choosing the optimal number of weights, there are various algorithms of Stochastic Gradient Descent but **_very efficient_** one which is **_Adam_** so that’s why we use Adam optimizer here. 
- The second parameter is **loss** this corresponds to loss function here we use **binary_cross-entropy** because if we see target attribute our dataset which contains the binary value that’s why we choose the binary cross-entropy. 
- The final parameter is metrics basically, it’s a list of metrics to be evaluated by the model and here we choose the accuracy metrics.

<center>
<img height="60" src="images\2.6.png" width="800"/>
</center>

####2.7 _Fitting the ANN_
- In this step we fit the training data our model X_train, y_train is our training data. 
- Here a **batch size** is basically **_a number of observations after which you want to update the weights_**. Here, we take batch size 10. 
- And the final parameter is **epoch** is basically **_when whole the training set passed through the ANN_**. Here, we choose the 100 number of the epoch.

<center>
<img height="60" src="images\2.7.png" width="700"/>
<br><br>
<img height="500" src="images\2.7_1.png" width="600"/>
<br><br>
<img height="500" src="images\2.7_2.png" width="600"/>
<br><br>
<img height="500" src="C:\images\2.7_3.png" width="600"/>
<br><br>
<img height="500" src="images\2.7_4.png" width="600"/>
<br><br>
<img height="500" src="images\2.7_5.png" width="600"/>
<br><br>
<img height="400" src="images\2.7_6.png" width="600"/>
</center>
                                                                                                                                                             
###3. Making the Prediction and Accuracy Result
####3.1 _Predict the test set Result_
- In this step, we predict our test set result.
- Here, our prediction results in probability, so we choose 1(customer leave the bank) if the probability is greater than one then 0.5 otherwise 0(customer don’t leave the bank).

<center>
<img height="100" src="is\3.1.png" width="400"/>
</center>

####3.2 _Confusion metrics_
- In this step, we make a **confusion metric** of our test set result. 
- To do that, we import **_confusion matrix_** from **_sklearn.metrics_**, then in confusion matrix, we pass two parameters first is **_y_test_** which is the actual test set result and second is **_y_pred_** which is the predicted result.

<center>
<img height="170" src="images\3.2.png" width="450"/>
</center>

####3.3 _Accuracy Score_
- In this step, we calculate the **accuracy score** based on the **_actual test result_** and **_predict test results_**.

<center>
<img height="150" src="images\3.3.png" width="420"/>

###So here we go we get _84.05%_ of our ANN model.
</center>
<br>

- In, the above implementation, we have used the **dense layer** in the first and second hidden layers structure.
- The model used is a sequential Model of **keras.models** which is used for the initialization of our ANN model.
- Below is the brief introduction about different keras models and its frequently used layers and few methods.

##Keras Models
There are three ways to create keras models.
- The **Sequential model**, which is very _**straightforward**_ (a simple list of layers), but is limited to single-input, single-output stacks of layers (as the name gives away).


- The **Functional API**, which is an **_easy-to-use_**, _fully-featured API_ that supports _arbitrary model_ architectures. This is the Keras **"industry strength"** model.


- **Model subclassing**, where _we implement everything from scratch on our own_. We normally use this if you have complex, out-of-the-box research use case.

###Model Training APIs
####compile method :
  - Configures the model for training.
  
```python
Model.compile(
    optimizer="rmsprop",
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    **kwargs
)
```

####fit method :
  - Trains the model for a fixed number of epochs (iterations on a dataset).
  - 
  
```python
Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
```
##Keras Layers
###Core Layers of Keras
####Input Function : 
```python
keras.Input(
    shape=None,
    batch_size=None,
    name=None,
    dtype=None,
    sparse=None,
    tensor=None,
    ragged=None,
    type_spec=None,
    **kwargs
)
```
  - Input() is used to instantiate a Keras tensor. 
  - A Keras tensor is a symbolic tensor-like object, which we augment with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model.

####Dense Layer :
```python
keras.layers.Dense(units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```
  - It is  regular densely-connected NN layer.
  - **Input shape** : N-D tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim). 
  - **Output shape** : N-D tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).

####Activation Layer
```python
keras.layers.Activation(activation, **kwargs)
```
  - Applies an activation function to an output.
  - The argument, **activation**, contains all the activation functions we have seen before. 
  - **Input shape** : Arbitrary. Use the keyword argument input_shape (tuple of integers, does not include the batch axis) when using this layer as the first layer in a model.
  - **Output shape** : Same shape as input.

####Embedding Layer
```python
keras.layers.Embedding(
    input_dim, output_dim,
    embeddings_initializer="uniform",
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    **kwargs
)
```
  - Turns positive integers (indexes) into dense vectors of fixed size.
  <br><br>
  
     e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
 <br><br>
  - This layer can only be used as the first layer in a model.
  - **Input shape** : 2D tensor with shape: (batch_size, input_length). 
  - **Output shape** : 3D tensor with shape: (batch_size, input_length, output_dim).
  
##Advantages of Artificial Neural Networks
- ANN is **_versatile_**, **_adaptive_** and **_scalable_**, _making them appropriate to tackle large datasets and highly complex **Machine Learning** tasks_ such as, 
  - Image Classification (eg., Google Images)
  - Speech Recognition (eg., Apple's Siri)
  - Video Recommendations (eg., YouTube)
  - Analyzing Sentiments among customers (eg., Twitter's Sentiment Analyzer)


- **Parallel Processing capability.**
  - Artificial neural networks have a numerical value that can perform more than one task simultaneously.


- **Storing data on the entire network.**
  - Data that is used in traditional programming is stored on the whole network, not on a database. 
  - The disappearance of a couple of pieces of data in one place doesn't prevent the network from working.


- **Capable to work with incomplete knowledge.**
  - After ANN training, the information may produce output even with inadequate data. 
  - The loss of performance here relies upon the significance of missing data.


- **Having memory distribution.**
  - For ANN is to be able to adapt, it is important to determine the examples and to encourage the network according to the desired output by demonstrating these examples to the network. 
  - The succession of the network is directly proportional to the chosen instances, and if the event can't appear to the network in all its aspects, it can produce false output.


- **Having fault tolerance.**
  - Extortion of one or more cells of ANN does not prohibit it from generating output, and this feature makes the network fault-tolerance.
  
##Disadvantages of Artificial Neural Network
- **Assurance of proper network structure.**
  - There is no particular guideline for determining the structure of artificial neural networks. 
  - The appropriate network structure is accomplished through experience, trial, and error.


- **Unrecognized behaviour of the network.**
  - It is the most significant issue of ANN. 
  - When ANN produces a testing solution, it does not provide insight concerning why and how. 
  - It decreases trust in the network.


- **Hardware dependence.**
  - Artificial neural networks need processors with parallel processing power, as per their structure. 
  - Therefore, the realization of the equipment is dependent.


- **Difficulty of showing the issue to the network.**
  - ANNs can work with numerical data. 
  - Problems must be converted into numerical values before being introduced to ANN. 
  - The presentation mechanism to be resolved here will directly impact the performance of the network. 
  - It relies on the user's abilities.


- **The duration of the network is unknown.**
  - The network is reduced to a specific value of the error, and this value does not give us optimum results.
  
##Applications of Neural Networks
They can perform tasks that are easy for a human but difficult for a machine:
- **Aerospace :**  Autopilot aircrafts, aircraft fault detection.


- **Automotive :** Automobile guidance systems.


- **Military :** Weapon orientation and steering, target tracking, object discrimination, facial recognition, signal/image identification.


- **Electronics :** Code sequence prediction, IC chip layout, chip failure analysis, machine vision, voice synthesis.


- **Financial :** Real estate appraisal, loan advisor, mortgage screening, corporate bond rating, portfolio trading program, corporate financial analysis, currency value prediction, document readers, credit application evaluators.


- **Industrial :** Manufacturing process control, product design and analysis, quality inspection systems, welding quality analysis, paper quality prediction, chemical product design analysis, dynamic modeling of chemical process systems, machine maintenance analysis, project bidding, planning, and management.


- **Medical :** Cancer cell analysis, EEG and ECG analysis, prosthetic design, transplant time optimizer.
 

- **Speech :** Speech recognition, speech classification, text to speech conversion.


- **Telecommunications :** Image and data compression, automated information services, real-time spoken language translation.


- **Transportation :** Truck Brake system diagnosis, vehicle scheduling, routing systems.


- **Software :** Pattern Recognition in facial recognition, optical character recognition, etc.


- **Time Series Prediction :** ANNs are used to make predictions on stocks and natural calamities.


- **Signal Processing :** Neural networks can be trained to process an audio signal and filter it appropriately in the hearing aids.


- **Control :** ANNs are often used to make steering decisions of physical vehicles.


- **Anomaly Detection :** As ANNs are expert at recognizing patterns, they can also be trained to generate an output when something unusual occurs that misfits the pattern.

- **Handwritten Character Recognition :** 
  - ANNs are used for handwritten character recognition. 
  - Neural Networks are trained to recognize the handwritten characters which can be in the form of letters or digits.
  

  <center>

  <img height="200" src="images\handwritten-character-recognition-Anns-applications.png" width="300"/>
  </center>

- **Speech Recognition :**
  - ANNs play an important role in speech recognition. 
  - The earlier models of Speech Recognition were based on statistical models like Hidden Markov Models. 
  - With the advent of deep learning, various types of neural networks are the absolute choice for obtaining an accurate classification.


  <center>

  <img height="200" src="images\ML-neural-networks-speech-recognition.png" width="300"/>
  </center>

- **Signature Classification :**
  - For recognizing signatures and categorizing them to the person’s class, we use artificial neural networks for building these systems for authentication. 
  - Furthermore, neural networks can also classify if the signature is fake or not.


  <center>

  <img height="200" src="images\ann-application-signature-classification.png" width="300"/>
  </center>

- **Facial Recognition :**
  - In order to recognize the faces based on the identity of the person, we make use of neural networks. 
  - They are most commonly used in areas where the users require security access. 
  - Convolutional Neural Networks are the most popular type of ANN used in this field.
  

  <center>

  <img height="300" src="images\image-recognition-in-neural-networks.png" width="300"/>
  </center>

##References
- https://www.javatpoint.com/artificial-neural-network
- https://data-flair.training/blogs/artificial-neural-networks-for-machine-learning/
- https://towardsdatascience.com/introduction-to-artificial-neural-networks-for-beginners-2d92a2fb9984
- https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_neural_networks.htm
- https://www.datasciencecentral.com/profiles/blogs/artificial-neural-network-ann-in-machine-learning
- https://www.analyticsvidhya.com/blog/2021/05/beginners-guide-to-artificial-neural-network/
- https://www.geeksforgeeks.org/activation-functions-neural-networks/
- https://en.wikipedia.org/wiki/Artificial_neural_network
- https://medium.com/machine-learning-researcher/artificial-neural-network-ann-4481fa33d85a
- https://keras.io/api/models/
- https://keras.io/api/layers/core_layers/
