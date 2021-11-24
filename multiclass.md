# MULTICLASS CLASSIFICATION OVERFITTING OR UNDERFITTING PROBLEMS

A classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multi-class classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time. 

In statistics a fit is referred to as, how close your model is to the target class / function / value. 

Most of the time, the cause of poor performance for a machine learning (ML) model is either **overfitting** or **underfitting**. A good model should be able to generalize and overcome both the overfitting and underfitting problems.

### What is Overfitting?
```markdown
Overfitting means that our ML model is modeling (has learned) the training data too well.
```
**Overfitting** referes to the situation where a model learns the data but also the noise that is part of **training data** to the extent that it **negatively impacts** the **performance** of the model on **new unseen data**.

In other worlds, the **noise** (i.e. random fluctuations) in the training set is **learned** as **rules/pattenrs** by the model. However, these noisy learned representations do not apply to new unseen data and thus, the model’s performance (i.e. accuracy, MSE, MAE) is negatively impacted.

Model is **overfitting** the training data when the model performs well on the training data but does not perform well on the evaluation data. That is, when the model’s error on the training set (i.e., during training) is very low but then, the model’s error on the test set (i.e., unseen samples) is large!

### What is Underfitting? 
```markdown
Underfitting means that our ML model can neither model the training data nor generalize to new unseen data.
```
A model that underfits the data will have poor performance on the training data. For example, in a scenario where someone would use a linear model to capture non-linear trends in the data, the model would underfit the data.

Model is **underfitting** the training data when the model performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples (often called X) and the target values (often called Y). That is, **Underfitting** is when the model’s error on both the training and test sets (i.e., during training and testing) is very high.

### How to limit Overfitting?
Actions that could limit overfitting are:
#### 1. Using **Cross Validation** Scheme.
   - For cross validation, best is to use **KFolds cross validation**. Using a KFolds scheme, we **train** and **test** our **model k-times** on different subsets of the **training data** and estimate a performance metric using the **test (unseen) data**.
    - Other options include the Leave-one-out cross-validation (LOOCV), the Leave-P-out cross-validation (LpOCV) and others.
   

#### 2. Reducing the **complexity** of the model. 
   - **Feature selection**: That is, consider using **fewer feature** combinations, **decrease n-grams size**, and **decrease the number of numeric attribute** bins.


### How to limit Underfitting?
- Poor performance on the training data could be because the model is too simple (the input features are not expressive enough) to describe the target well. Performance can be improved by increasing model flexibility. To increase model flexibility, try the following:
  - Add new domain-specific features and more feature Cartesian products, and change the types of feature processing used (e.g., increasing n-grams size)
  - Decrease the amount of regularization used.


- Accuracy on training and test data could be poor because the learning algorithm did not have enough data to learn from. You could improve performance by doing the following:
  - Increase the amount of training data examples.
  - Increase the number of passes on the existing training data.
<br><br>

**Underfitting** or **overfitting** problem can also be because of the **imbalance data** in the dataset.

## What is Imbalanced Dataset?
- **Imbalanced data** typically refers to a **problem** with classification problems where the **classes are not represented equally**. 
- **For example**, we may have a 3-class classification problem of set of fruits to classify as oranges, apples or pears with total 100 instances . A total of 80 instances are labeled with Class-1 (Oranges), 10 instances with Class-2 (Apples) and the remaining 10 instances are labeled with Class-3 (Pears). This is an **imbalanced dataset** and the **ratio** of **8:1:1**. 
- Most classification data sets do not have exactly equal number of instances in each class, but a small difference often does not matter. There are problems where a class **imbalance** is not just common, it is **expected**. 
- **For example**, in datasets like those that characterize fraudulent transactions are imbalanced. The vast majority of the transactions will be in the “Not-Fraud” class and a very small minority will be in the “Fraud” class.

### How to measure model on imbalanced data performance? 
- Let us consider that we train our model on imbalanced data of earlier example of fruits and since the data is **heavily biased** towards Class-1 (Oranges), the model over-fits on the Class-1 label and predicts it in most of the cases, and we achieve an **accuracy** of **80%** which **seems very good at first** but looking closely, it may never be able to classify apples or pears correctly. 
- Now the question is if the accuracy, in this case, is not the right metric to choose then **_What metrics to use to measure the performance of the model?_**

### Confusion-Matrix
- With **imbalanced classes**, it’s easy to get a high accuracy without actually making useful predictions. 
- So, **accuracy** as an **evaluation metrics** makes sense only if the class labels are uniformly distributed. In case of **imbalanced classes** **confusion-matrix** is good technique to summarizing the performance of a classification algorithm.

```markdown
Confusion Matrix is a performance measurement for a classification algorithm where output can be two or more classes.
```

Looking at the confusion matrix one can clearly see how the model is performing on classifying various classes.

## How to improve the performance?
There are various techniques involved in improving the performance of imbalanced datasets.

### Re-sampling Dataset
To make our dataset balanced there are two ways to do so:
1. **Under-sampling :** Remove samples from over-represented classes ; use this if you have huge dataset.
2. **Over-sampling :** Add more samples from under-represented classes; use this if you have small dataset.

### SMOTE (Synthetic Minority Over-sampling Technique)
**SMOTE** is an **over-sampling** method. It creates synthetic samples of the minority class. We use **imblearn python package** to over-sample the minority classes .