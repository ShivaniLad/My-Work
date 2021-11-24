# IMBALANCED-LEARN LIBRARY FOR IMBALANCED DATASETS

There are two Resampling methods:
1. Over-sampling methods
2. Under-sampling methods

## UNDER SAMPLING METHODS
The **`imblearn.under_sampling`** provides methods to under-sample a dataset.

### Prototype generation method
The **`imblearn.under_sampling.prototype_generation`** submodule contains methods that generate new samples in order to balance the dataset.

### ClusterCentroid
```python
imblearn.under_sampling.ClusterCentroids(sampling_strategy='auto', 
                                         random_state=None, 
                                         estimator=None,
                                         voting='auto',
                                         n_jobs='deprecated')
```
Undersample by generating centroids based on clustering methods.

Method that under samples the majority class by **replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm**. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.

### Prototype selection methods

The **`imblearn.under_sampling.prototype_selection`** submodule contains methods that select samples in order to balance the dataset.

### 1. CondensedNearestNeighbour
```python
imblearn.under_sampling.CondensedNearestNeighbour(sampling_strategy='auto', 
                                                  random_state=None, 
                                                  n_neighbors=None, n_seeds_S=1, 
                                                  n_jobs=None)
```
Undersample based on the condensed nearest neighbour method.

**CondensedNearestNeighbour** uses a **1 nearest neighbor** rule to iteratively decide if a sample should be removed or not. The algorithm is running as followed:

1. Get all minority samples in a set C.

2. Add a sample from the targeted class (class to be under-sampled) in C and all other samples of this class in a set S.

3. Go through the set S, sample by sample, and classify each sample using a 1 nearest neighbor rule.

4. If the sample is misclassified, add it to C, otherwise do nothing.

5. Reiterate on S until there is no samples to be added.

**What is 1-nearest neighbor classifier ?** 

As the size of training data set approaches infinity, the one nearest neighbour classifier **guarantees an error rate of no worse than twice the Bayes error rate** (the minimum achievable error rate given the distribution of the data).

### 2. EditedNearestNeighbours
```python
imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy='auto', 
                                                n_neighbors=3, kind_sel='all', 
                                                n_jobs=None)
```
Undersample based on the edited nearest neighbour method.
<br>
This method will clean the database by removing samples close to the decision boundary.

**EditedNearestNeighbours** applies a **nearest-neighbors algorithm** and “**edit**” the dataset **by removing samples which do not agree “enough”** with their neighborhood. For each, sample in the class to be under-sampled, the nearest-neighbours are computed and if the selection criterion is not fulfilled, the sample is removed.

Two selection criteria are currently available: 
1. **The majority** (i.e., kind_sel='mode') 
2. **All** (i.e., kind_sel='all') 

The nearest-neighbors have to belong to the same class as the sample inspected to keep it in the dataset. Thus, it implies that **kind_sel='all' will be less conservative than kind_sel='mode'**, and more samples will be excluded in the former strategy than the latest.

### 3. RepeatedEditedNearestNeighbours

```python
imblearn.under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy='auto', 
                                                        n_neighbors=3, max_iter=100,
                                                        kind_sel='all', n_jobs=None)
```
Undersample based on the repeated edited nearest neighbour method.
<br>
This method will repeat several time the ENN algorithm.

**RepeatedEditedNearestNeighbours extends EditedNearestNeighbours by repeating the algorithm multiple times**. 
<br>
Generally, repeating the algorithm will delete more data.

### 4. AIIKNN
```python
imblearn.under_sampling.AllKNN(sampling_strategy='auto', n_neighbors=3,
                               kind_sel='all', allow_minority=False, 
                               n_jobs=None)
```
Undersample based on the AllKNN method.
<br>
This method will **apply ENN several time** and **will vary the number of nearest neighbours**.

**AllKNN differs** from the previous **RepeatedEditedNearestNeighbours** since the **number of neighbors of the internal nearest neighbours algorithm is increased at each iteration**.

### 5. InstanceHardnessThreshold
```python
imblearn.under_sampling.InstanceHardnessThreshold(estimator=None,
                                                  sampling_strategy='auto',
                                                  random_state=None, cv=5, 
                                                  n_jobs=None)
```

Undersample based on the instance hardness threshold.

**InstanceHardnessThreshold** is a specific algorithm in which a **classifier is trained on the data** and the **samples with lower probabilities are removed**.

This class has **2 important parameters**. 
- The **estimator** will accept any scikit-learn classifier which has a method **predict_proba**. The classifier training is performed using a cross-validation.
- The **parameter cv** can set the number of folds to use.

_**InstanceHardnessThreshold** could almost be considered as a controlled under-sampling method. However, due to the probability outputs, it is not always possible to get a specific number of samples._

### 6. NearMiss
```python
imblearn.under_sampling.NearMiss(sampling_strategy='auto', version=1, 
                                 n_neighbors=3, n_neighbors_ver3=3, n_jobs=None)
```
Class to perform under-sampling based on NearMiss methods.

**NearMiss adds some heuristic rules to select samples**. NearMiss implements 3 different types of heuristic which can be selected with the parameter **version**.

### 7. NeighbourhoodCleaningRule
```python
imblearn.under_sampling.NeighbourhoodCleaningRule(sampling_strategy='auto',
                                                  n_neighbors=3, kind_sel='all',
                                                  threshold_cleaning=0.5,
                                                  n_jobs=None)
```
Undersample based on the neighbourhood cleaning rule.
<br>
This class **uses ENN and a k-NN to remove noisy sample**s from the datasets.

**NeighbourhoodCleaningRule** will **focus on cleaning the data than condensing them**. Therefore, it **will use the union of samples to be rejected between the EditedNearestNeighbours and the output a 3 nearest neighbors classifier**.

### 8. OneSidedSelection
```python
imblearn.under_sampling.OneSidedSelection(sampling_strategy='auto',
                                          random_state=None, n_neighbors=None,
                                          n_seeds_S=1, n_jobs=None)
```

Class to perform under-sampling based on one-sided selection method.

In the contrary to CondensedNearestNeighbour, **OneSidedSelection will use TomekLinks to remove noisy samples**. In addition, the **1 nearest neighbor rule is applied to all samples** and the one **which are misclassified will be added to the set C**. No iteration on the set S will take place.

### 9. RandomUnderSampler
```python
imblearn.under_sampling.RandomUnderSampler(sampling_strategy='auto',
                                           random_state=None, replacement=False)
```
Class to perform random under-sampling.
<br>
_**Under-sample the majority class(es) by randomly picking samples with or without replacement.**_

Supports multi-class resampling by sampling each class independently. **Supports heterogeneous data** as object array containing string and numeric data.

**RandomUnderSampler** is a **fast and easy** way to **balance the data** by **randomly selecting** a subset of **data** for the targeted classes.

**RandomUnderSampler** allows to bootstrap the data by setting **replacement to True**. The resampling with multiple classes is performed by considering independently each targeted class.

### 10. TomekLinks
```python
imblearn.under_sampling.TomekLinks(sampling_strategy='auto', n_jobs=None)
```
Under-sampling by removing Tomek’s links.

TomekLinks detects the so-called Tomek’s links. A Tomek’s link between two samples of different class x and y is defined such that for any sample z:

**d(x, y) < d(x, z) \text{ and } d(x, y) < d(y, z)**

where, **_d(.) is the distance between the two samples_**. In some other words, a Tomek’s link exist if the two samples are the nearest neighbors of each other.

In the figure below, a Tomek’s link is illustrated by highlighting the samples of interest in green.

<img height="400" src="C:\Users\shivani.lad\Desktop\documentation\images\illustration_tomek_links_001.png" width="450"/>

_The parameter sampling_strategy control which sample of the link will be removed_. For instance, the default (i.e., sampling_strategy='auto') will remove the sample from the majority class. Both samples from the majority and minority class can be removed by setting sampling_strategy to 'all'.

 The figure illustrates this behaviour.

<img height="400" src="C:\Users\shivani.lad\Desktop\documentation\images\illustration_tomek_links_002.png" width="600"/>


## Over-sampling methods

The **`imblearn.over_sampling`** provides a set of method to perform over-sampling.

### Basic Over-Sampling : RandomOverSampler
```python
imblearn.over_sampling.RandomOverSampler(sampling_strategy='auto', random_state=None,
                                         shrinkage=None)
```
Class to perform random over-sampling.

Object to over-sample the minority class(es) by **picking samples at random with replacement**. The bootstrap(loading a small initial program) can be generated in a smoothed manner.

**One way** to fight this issue is to **generate new samples in the classes which are under-represented**. The most naive strategy is to **generate new samples by randomly sampling with replacement** the current available samples. The RandomOverSampler offers such scheme.

In the figure below, we compare the decision functions of a classifier trained using the over-sampled data set and the original data set.

<img height="400" src="C:\Users\shivani.lad\Desktop\documentation\images\comparison_over_sampling_002.png" width="600"/>

As a result, the _**majority class does not take over the other classes during the training process**_. Consequently, all classes are represented by the decision function.

In addition, **RandomOverSampler** allows sampling **heterogeneous data** (e.g. containing some strings).

If **repeating samples** is an **issue**, the _parameter **shrinkage**_ **allows** to **create** a **smoothed bootstrap**. 

However, the **original data needs to be numerical**. The **shrinkage** parameter **controls** the **dispersion of the new generated samples**. Below figure shows that the new samples are not overlapping anymore once using a smoothed bootstrap. This ways of generating smoothed bootstrap is also known a **Random Over-Sampling Examples (ROSE)**

<img height="400" src="C:\Users\shivani.lad\Desktop\documentation\images\comparison_over_sampling_003.png" width="600"/>

**Supports multi-class resampling** by sampling **each** class **independently**.
<br>
Since smoothed bootstrap are generated by adding a small perturbation to the drawn samples, this method is not adequate when working with sparse matrices.

### SMOTE Algorithms

<img height="400" src="C:\Users\shivani.lad\Desktop\documentation\images\7_Smote_techniques.jpeg" width="600"/>

Apart from the random sampling with replacement, there are two popular methods to over-sample minority classes: 
1. The Synthetic Minority Oversampling Technique (SMOTE)
2. The Adaptive Synthetic (ADASYN) sampling method.

There are different **variations** of **SMOTE**.

### 1. SMOTE
```python
imblearn.over_sampling.SMOTE(sampling_strategy='auto',
                             random_state=None, k_neighbors=5,
                             n_jobs=None)
```
Class to perform over-sampling using SMOTE.

**SMOTE** stands for _**Synthetic Minority Over-sampling Technique**_. It creates new synthetic samples to balance the dataset.

In the case of random oversampling, it was prone to overfit as the minority class samples are replicated, here SMOTE comes into the picture.

**SMOTE** **works** by **utilizing** a **k-nearest neighbor algorithm** to create synthetic data. **Steps** samples are created using Smote:
- Identify the feature vector and its nearest neighbor 
- Compute the distance between the two sample points 
- Multiply the distance with a random number between 0 and 1. 
- Identify a new point on the line segment at the computed distance. 
- Repeat the process for identified feature vectors.

### 2. SMOTENC

```python
imblearn.over_sampling.SMOTENC(categorical_features,
                               sampling_strategy='auto',random_state=None,
                               k_neighbors=5, n_jobs=None)
```
**Synthetic Minority Over-sampling Technique (SMOTE)** for **Nominal** and **Continuous**.

Unlike SMOTE, SMOTE-NC for dataset containing numerical and categorical features. However, it is not designed to work with only **categorical features.** 
 
Smote can also be used for data with categorical features, by **one-hot encoding** but it may **result** in an **increase** in **dimensionality**. 
<BR>
**Label Encoding** can also be used to convert categorical to numerical, but **after smote** it may **result** in **unnecessary information**. 
<BR>
This is why we need to use **SMOTE-NC** when we have cases of **mixed data**. 

**_**Smote-NC** can be used by denoting the **features** that **are** **categorical**, and **Smote** would **resample** the categorical data instead of creating synthetic data._**

### 3. SMOTEN

```python
imblearn.over_sampling.SMOTEN(sampling_strategy='auto', random_state=None,
                              k_neighbors=5, n_jobs=None)
```
Synthetic Minority Over-sampling Technique for Nominal.
<br>
It expects that the **data** to **resample** are **only** made of **categorical features.**

The algorithm changes in two ways:
- The nearest neighbors search does not rely on the Euclidean distance. Indeed, the **Value Difference Metric (VDM)** also implemented in the class **_ValueDifferenceMetric_** is used. 
- A new sample is generated where each feature value corresponds to the most common category seen in the neighbors samples belonging to the same class.

### 4. BorderlineSMOTE
```python
imblearn.over_sampling.BorderlineSMOTE(sampling_strategy='auto',
                                       random_state=None, k_neighbors=5,
                                       n_jobs=None, m_neighbors=10,
                                       kind='borderline-1')
```
Over-sampling using Borderline SMOTE.
<br>
Borderline samples will be detected and used to generate new synthetic samples.

Due to the presence of some minority points or outliers within the region of majority class points, bridges of minority class points are created. This is a problem in the case of Smote and is solved using Borderline Smote.

In the Borderline Smote technique, only the minority examples near the borderline are over-sampled. It classifies the minority class points into noise points, border points. 

Noise points are minority class points that have most of the points as majority points in its neighbor, and border points have both majority and minority class points in its neighbor. Borderline Smote algorithm tries to create synthetic points using only these border points and ignore the noise points.

### 5. KMeansSMOTE
```python
imblearn.over_sampling.KMeansSMOTE(sampling_strategy='auto', random_state=None,
                                   k_neighbors=2, n_jobs=None,
                                   kmeans_estimator=None,
                                   cluster_balance_threshold='auto',
                                   density_exponent='auto')
```
Apply a KMeans clustering before to over-sample using SMOTE.
 
**KMeansSMOTE** uses a _**KMeans clustering method before to apply SMOTE**_. The clustering will group samples together and generate new samples depending on the cluster density.

K-Means SMOTE is an oversampling method for class-imbalanced data. It aids classification by generating minority class samples in safe and crucial areas of the input space. The method avoids the generation of noise and effectively overcomes imbalances between and within classes.

**K-Means SMOTE** works in following **steps**:
- Cluster the entire data using the k-means clustering algorithm. 
- Select clusters that have a high number of minority class samples.
- Assign more synthetic samples to clusters where minority class samples are sparsely distributed.

### 6. SVMSMOTE
```python
imblearn.over_sampling.SVMSMOTE(sampling_strategy='auto', random_state=None,
                                k_neighbors=5, n_jobs=None, m_neighbors=10,
                                svm_estimator=None, out_step=0.5)
```
Over-sampling using **SVM-SMOTE**.

**Variant of SMOTE** algorithm which **use** an **SVM algorithm** to detect sample to use for generating new synthetic samples.

SVM-SMOTE uses an SVM classifier to find support vectors and generate samples considering them. Note that the C parameter of the SVM classifier allows selecting more or less support vectors.

For both borderline and SVM SMOTE, a neighborhood is defined using the parameter m_neighbors to decide if a sample is in danger, safe, or noise.

### Adaptive Synthetic Sampling - ADASYN
```python
imblearn.over_sampling.ADASYN(sampling_strategy='auto', random_state=None,
                              n_neighbors=5, n_jobs=None)
```
Oversample using **Adaptive Synthetic (ADASYN)** algorithm.

This method is **similar to SMOTE,** **but** it _generates different number of samples depending on an estimate of the local distribution of the class to be oversampled._

ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier.

Borderline Smote gives more importance or creates synthetic points using only the extreme observations that are the border points and ignores the rest of minority class points. This problem is solved by the ADASYN algorithm, as it **creates synthetic data according to the data density**.

**_The synthetic data generation is inversely proportional to the density of the minority class_**. 

A comparatively larger number of synthetic data is created in regions of a low density of minority class than higher density regions.

In other terms, **_in the less dense area of the minority class, the synthetic data are created more._**

_**Supports multi-class resampling. A one-vs.-rest scheme is used.**_

## Methods

**fit(X, y) :** Check inputs and statistics of the sampler.

**fit_resample(X, y) :** Resample the dataset.

**get_params([deep]) :** Get parameters for this estimator.

**set_params**(****params) :** Set the parameters of this estimator.