# Different Encoding Techniques for Categorical Data

- Most of the Machine Learning Algorithms cannot handle categorical data until they are converted into numerical data.
- Many algorithm’s performances vary based on how Categorical variables are encoded.
- The categorical data's are basically divided into 2 categories.
  1. **Nominal** : Data which doesn't have any particular order.
  2. **Ordinal** : Data which are in or has some order.

<center>
<img height="400" src="categorical data.png" width="600"/>
</center>

- ### Examples for the **Nominal** variable:
  - **Colors** : Red, Yellow, Pink, Blue
  - **Country** : Singapore, Japan, USA, India, Korea
  - **Animal** : Cow, Dog, Cat, Snake
  - **Gender** : Male, Female
  
- ### Example of **Ordinal** variables:
  - High, Medium, Low
  - Strongly agree, Agree, Neutral, Disagree, and Strongly Disagree.
  - Excellent, Okay, Bad

- There are many ways we can encode these categorical variables as numbers and use them in an algorithm.
  1. One Hot Encoding
  2. Label Encoding
  3. Ordinal Encoding
  4. Helmert Encoding
  5. Binary Encoding
  6. Frequency Encoding
  7. Mean Encoding
