# MIDAS Summer Internship Program Task-3
This repository contains a solution for the MIDAS Summer Internship Task-3: Product Category Prediction using the Flipkart E-Commerce Dataset

## Problem Statement:

Using the Flipkart E-Commerce dataset, build a model to predict the product category with product description as the primary feature. Explain the data preprocessing, data visualization steps along with the model evaluation criteria. 

## Solution:
 

### Primary category selection:

- To decide the primary product category for classification, I first separated all the product categories from the “category-tree” column. A visualization of the class-distributions in each category level is described using the pie-charts below. 
- The highest coverage (number of records) can be seen in category level 1.  All the rest of the category levels contain a varying number of missing values (NaN).

 Category Level 1     |   Category Level 3 |  Category Level 3
:-------------------------:| :-------------------------: |:-------------------------:|
<img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category1.png" width="350" height = "350" /> | <img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category2.png" width="350" height = "350" /> | <img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category3.png" width="350" height = "350" /> |

 Category Level 4     |   Category Level 5 |  Category Level 6
:-------------------------:| :-------------------------: |:-------------------------:|
<img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category4.png" width="350" height = "350" /> | <img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category5.png" width="350" height = "350" /> | <img src = "https://github.com/AmeyHengle/MIDAS-Task3/blob/main/visualization%20plots/category6.png" width="350" height = "350" /> |

### Data preparation:

- The primary category consists of a large number of class labels (265) with varying frequency. 
In order to focus on relevant classes, I only consider class labels that constitute at least 0.5 percent of the total training set (20,000). 
- Thus, the final dataset is reduced to 19,287 records spanning 18 class labels. 
This dataset is further split into training and validation sets, using a standard stratified 80:20 split. 

### Text cleaning and normalization:

- The text data (description) is preprocessed by eliminating urls, non-printable characters, missing delimiters, letter repetitions, non-word repetitions, informal parentheses, phrase repetition and noisy text.
- After the basic preprocessing steps, the data is subjected to a further word-level normalization using typo-fixing and stemming. 
An overview of all the preprocessing steps can be found in preprocess.py

### Feature Engineering:

Following feature-extraction techniques are used to convert the description text to a fixed-dimensional input vector:

- Word Frequency (BoW) using unigrams, bigrams and trigrams. 
- Word Frequency (BoW) using bigrams.
- Word Frequency (BoW) using trigrams. 
- TF-IDF using unigrams, bigrams and trigrams. 
- TF-IDF using bigrams.
- TF-IDF using trigrams. 


### Models:

I have explored multiple standard machine learning models for the given task. This includes Naive Bayes, Random Forest, K Nearest Neighbor, SVM and Linear SCV classifiers. 

### Evaluation Criteria:

The class-distribution diagram indicates a large disparity between the class labels. Thus using metrics like accuracy can be misleading here, as it overlooks the data-imbalance problem. Hence, I use the “macro-f1” score as the metric to evaluate a model’s performance, to ensure that the model is penalized for not performing well on the minority classes. 


### Result:

A brief overview of the top 10 models (based on macro-f1 score) is given in the table below. 
The best results are obtained using Linear SVM model with n-grams as input features. 

| Model                    | Input Feature | accuracy    | macro-f1    | macro-precision | macro-recall | weighted-f1 | weighted-precision | weighted-recall |
|--------------------------|---------------|-------------|-------------|-----------------|--------------|-------------|--------------------|-----------------|
| Linear SVC               | tfidf_ngram   | 97.82270607 | 0.958705114 | 0.969025295     | 0.949900225  | 0.977997811 | 0.978329609        | 0.978227061     |
| Linear SVC               | bow_bigram    | 97.74494557 | 0.957083574 | 0.962276837     | 0.952640219  | 0.977227599 | 0.977346629        | 0.977449456     |
| Linear SVC               | bow_trigram   | 96.78589943 | 0.945856146 | 0.959168167     | 0.935118328  | 0.967379315 | 0.968175178        | 0.967858994     |
| KNN Classifier           | tfidf_ngram   | 96.91550026 | 0.942768726 | 0.944196273     | 0.94215568   | 0.968954882 | 0.969089297        | 0.969155003     |
| SVM Classifier           | bow_ngram     | 97.04510109 | 0.942510983 | 0.948931997     | 0.93741427   | 0.970213287 | 0.970545628        | 0.970451011     |
| Random Forest Classifier | bow_bigram    | 96.6562986  | 0.942123211 | 0.957264745     | 0.930930284  | 0.965751805 | 0.967065149        | 0.966562986     |
| KNN Classifier           | tfidf_bigram  | 96.44893727 | 0.940968195 | 0.944647358     | 0.937896408  | 0.964330975 | 0.964529407        | 0.964489373     |
| SVM Classifier           | bow_bigram    | 96.52669777 | 0.940859664 | 0.954512726     | 0.928566046  | 0.964890014 | 0.965349642        | 0.965266978     |
| SVM Classifier           | tfidf_ngram   | 96.60445827 | 0.938320163 | 0.964201242     | 0.917127838  | 0.96538514  | 0.966747161        | 0.966044583     |
| Random Forest Classifier | bow_ngram     | 96.39709694 | 0.937666653 | 0.958027051     | 0.921570304  | 0.963081731 | 0.964691526        | 0.963970969     |




### Discussion and Conclusion:


- All the machine learning models perform considerably well for the given task, showing consistent performance even for minority classes. 

- Hence, this limits the immediate need to employ deep learning variants like CNN, LSTM or complex models like BERT for the given classification task. 

- For further improving the classification performance, we can employ the ensemble techniques like Stacking, Blending, Bagging and Boosting using the current set of models. 
We can explore more advanced ensemble algorithms like AdaBoost and GBM. 

- For better generalization, we can design a hybrid neural network architecture to incorporate the textual as well as image representation of a product in a single model. 




