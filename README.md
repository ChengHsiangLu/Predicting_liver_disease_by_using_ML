## Title
**Predicting liver disease by using machine learning models**

## Author
**Cheng-Hsiang Lu**

## Instructor
**Dr. Baloglu**


## Introduction

The liver is the largest solid organ in the human body. It's located mainly in the upper right portion of your abdomen, beneath the diaphragm and above your stomach. The liver carries out over 500 essential tasks. The major functions of liver include: metabolizing bilirubin, fat, and proteins, supporting blood clots, Immunological function, producing albumin, etc. The term “liver disease” refers to any of several conditions that can damage your liver. Over time, liver disease can cause cirrhosis. As more scar tissue replaces healthy liver tissue, the liver can no longer function properly. If left untreated, liver disease can lead to liver failure and liver cancer.

## Dataset

My dataset was found on Kaggle. It contains more than 30 thousand data points and 11 features. There are 10 features which are Age, Gender, total Bilirubin, direct Bilirubin, ALP, GPT, GOT, Total Proteins, Albumin, and AG_Ratio. When it comes to liver, expect for Age and Gender, these are the most commonly performed blood tests. There is one target that has two classes: 0 correspond to sample without liver disease and 1 correspond to sample with liver disease.

## Goal

The goal for this project is to predict liver disease by using these data points and features.

## Missing values

First, I checked my dataset to see if there are missing values. Unfortunately, All features contain missing values. Then, I decided whether I should remove them. I tried to remove all of them, but 12% of my dataset would disappear. Therefore, I change to replace all missing values with means or median.

## Mean or median

From histograms and my description table, I can see that histograms of Age, T_P, ALB, and AG_Ratio look like a normal distribution. So, we can replace their missing values with means. On the other hand, histograms of T_B, D_B, ALP, GPT, GOT look like a skewed distribution. Thus, we can replace their missing value with medians.

![](/Pictures/Histogram_mean_median.png?raw=true)

After that, only the gender feature contain missing values. However, I cannot replace them with mean or median because gender is a categorical data. But, based on these missing values of Gender are less than 3% of all data points, I just remove them.

## Imbalanced dataset

I take a closer look at my dataset’s target distribution and find out that my dataset is imbalanced. It shows that more than 21 thousand data points have liver disease, but only 8 thousand data points with no liver disease. There is a gap between two groups. The gap could cause inaccuracy for my further analysis. As the result, I decide to reduce the data points in liver disease group to match the number in no liver disease group.

## Data split and merge

First, I split my original data into two group by Target (0 and 1).

Second, In the data with liver disease, I use pandas sample function to randomly extract 40% of the data. There are about 8000 data points are extracted.

Then, I merge data with and without liver disease two group into one new dataset.

## Balanced dataset

After reorganizing them, I visualize my data to see if there are any changes.

From the bar plot, It looks like a much more balance dataset.

And I recheck their distributions which look similar to the former ones.

There are about 8,000 data points within each group now.

## Correlations

Next thing I want to check correlations between features before I analyze my data. Maybe there are some redundant features that I would like to remove or essential features that I want to keep. In the end, It will improve the performance of my models.

### Pairplot

First, a pairplot allows us to see both distribution of single variables and relationships between two variables. Orange dots represent samples with no liver disease and blue dots represent samples with liver. From the pair plot, I can see the linear patterns between T_B and D_B. It suggest a presence of strong correlation between these two features. Therefore, I only have to pick one of the them when I go into further analysis.

### Heatmap

In this heatmap, a greater correlation shows a darker color and a higher score. There are four areas in blue squares show strong correlations. The feature of T_B and D_B, GPT and GOT, T_P and ALB, the last one is ALB and  AG_Ratio. In the end, I'm going to remove three features: T_B, GPT, and ALB.

## Drop unnecessary features

After dropping these unnecessary features, there are less obvious linear patterns in the pairplot. The heatmap shows no strong correlation between each feature as well. Initially, I have 10 features. After checking their correlations, I decide to move on with these 7 features which are Age, Gender, D_B, ALP, GOT, T_P, and AG_Ratio. Next, I’m going to build my model.

## Training and test dataset

The training dataset is a initial data used to train machine learning models. They’re fed to machine learning algorithms to teach them how to make predictions. The test dataset is a secondary dataset which is used to test machine learning models after is has been trained on a training dataset. Generally, test dataset is used to test your model to see if it is working properly or not. Training and test datasets must not have any overlap, so that test datasets can really measure the capabilities of your models. Normally, people suggest that 70% of the available data is allocated for training set and 30% is allocated for test set. 

## Standardize the Data

First, I have to standardize my dataset. Fit is the function used on the training dataset only. In order to find the mean, variance, maximum and minimum values of the training dataset, the inherent properties of the training dataset. Transform is the function used on both the training and test dataset. The reason why we use transform function is to scale the data normalize and standardize them.

## Confusion matrix

Now, I’m going to talk about the confusion matrix. The  x-axis shows a predicted label and the y-axis shows a true label. We have positive and negative under our labels. 

The upper left corner presents the true negative value. Just like people with no liver diseases and were predicted negative. 

The lower right corner presents the true positive value. Like people with liver diseases and were predicted positive.

The upper right corner presents the false positive value. Like people with no liver diseases and were incorrectly predicted positive.

The lower left corner presents the false negative value. Like people with liver diseases and were incorrectly predicted negative.

To calculate the accuracy of the confusion matrix, there is a equation right here. First we add up TN and TP and then divided them by all four parts will get the accuracy score.

## Models

### Logistic Regression with a single feature

The first model I used is Logistic Regression model. Logistic regression is a process of modeling the probability. The most common logistic regression model is based on binary outputs. For example, like true/false, yes/no, and so on. It is a useful analysis method for the classification problem which is what I have. First, I split my dataset into training and test datasets with training size 70% and test size 30%. Then, I try a single feature by picking a variable (D_B) based on the heatmap score.

The accuracy of test dataset is 66% and the training dataset accuracy is 67%. It is not the ideal score that I want because I only use one feature. Now, I’m going to use logistic regression with all feature in my dataset and build a more complicate model.

### Logistic Regression with all features

When I use logistic regression with all features, the accuracy of test dataset increases from 66% to 71% and the training dataset accuracy also increases from 67% to 71%. However, it is still not the best score. Therefore, I’m going to find out the best model by comparing different models all together.

## Find best model

This time, I’ll compare logistic regression, decision tree, and Random forest. 

### Decision tree

The decision tree is a supervised machine learning model. It is like a tree, branching from the root to the trunk to the leaves.  The part of each node is a feature. A step-by-step breakdown of features will yield outputs on whether patients have liver disease or not.

### Random forest

Random forest is a supervised machine learning algorithm that is used widely in classification and regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression. For example, in this figure, there are total 9 decision trees, 6 of them predict 1 and 3 of them predict 0. So, the most votes 1 becomes the model’s prediction.

## Best model

After checking with these three models, I found that the random forest model has the highest accuracy. Its accuracy is 88% which is higher than logistic regression and decision tree. I can also see that decision tree and random forest have a much higher score than the logistic regression. But, it’s known that with tree-based models, the score I get may be due to a so-called overfitting problem. I have to double check whether the relatively high score is due to the overfitting issue.

## Overfitting

Overfitting is over-learning the training data, and it becomes hard to predict other data that is not in the training dataset. Any model perform too well on the training dataset but the performance drops significantly over the test dataset.

As the model complexity increases, we can reduce the training error but simultaneously increases the test error.

## Cross validation

By using cross validation, we can detect overfitting problem. Cross validation is a technique for evaluating machine learning models by training several models on subsets of the available input data and evaluating them on the complementary subset of the data. There are different types of Cross Validation Techniques, but the overall concept remains the same.

The first step is to partition the data into a number of subsets, in this case, there are 9 subsets.

Second, Hold out a set at a time and train the model on remaining set, the set in yellow is the hold-out set.

Third, test the model on hold-out set.

The last step is to repeat the process for each subset of the dataset.

In the end, my average of Accuracy is 88%.

## Model Evaluation

After using cross validation, I evaluate the performance of my model. The accuracy of test dataset is 90% and the accuracy of training dataset is 91.8%. The higher score that I got from the training dataset is almost 4% higher than the Average Accuracy of cross validation

## Feature importance

The next thing I would like to do is to rank my features and see which one is more important. I use random forest to rank my features. The reason why I want to do that is to see if some features are not that important, then I can remove them and improve my accuracy. We can see that ALP, GOT, D__B, T_P, and AG_Ratio are the top five important features in my dataset. On the other hand, Age and Gender are not that essential. Thus, I decide to remove these two features.

## Remove age and gender

After removing the feature age and gender, the cross validation average Accuracy slightly increases to 91% and the accuracy of test dataset is still around 90%. Last, the accuracy of training dataset slightly increases to 94%. So, by removing age and gender, my training dataset accuracy is improved. Then, I’m going to use this model to predict new data which is obtain from my former co-work who currently work in the hospital.

## Make predictions

The first two samples’ blood tests are obtained from my co-workers themselves who are both healthy and the last two samples’ blood tests are gained from people with slightly abnormal liver function, but they still consider healthy. However, all results show that they might have liver disease.

## Possible reasons

I have listed three common possible reasons for incorrect predictions.

First, the most common reason for incorrect prediction is the imbalanced dataset. To be more specific, if one group is 100 fold greater than the other group, it could be possible that my model only select one certain group to go through the train and test process. So, the training and test scores are always high but didn’t work well with data from other dataset. However, I have modified my dataset with the same amount of data points in two groups. Therefore, it is not likely to be the reason.


Second, it could be the overfitting problem. I found that some people discuss in stack over flow mention that the best models have high accuracy on training data and equally high accuracy on test data. Plus, both accuracy metrics are not more than 5~10% of each other, which shows model stability. The lower difference the better. So I quickly review my test and training accuracy score. The test dataset accuracy is 90% and the training dataset accuracy is 94%. Although the accuracy of my model looks like it fit the definition of a good model, a simple definition of overfitting is “when a model is no longer as accurate as we want it to be on data we care about.” So, here comes the last possible reason.


If you have duplicates in your training and test datasets, it’s expected to have high accuracies. When you have duplicates in the training dataset, it is effectively the same as having its 'weight' doubled. That element becomes twice as important when the classifier is fitting your data and the classifier becomes biased towards correctly classifying that particular scenario over others. When you have duplicates in the test dataset, you'll get an inflated sense of how well the model performs overall. Because the rarer scenarios are less well represented, and the classifier's poor performance with them will contribute less to the overall test score.


## Duplicates

After removing Age and Gender, within different groups, all features are showed exact the same. It might cause the model keep learning from the same training data and increasing the weight. It also inflated the score of test dataset accuracy due to these duplicates. However, the model have not encountered any new data with different values. In the end, it predict incorrectly.


## Normal range

From my personal point of view, There is another possible reason which is the different normal range of each feature. For example, the figure below is the description of non-liver disease group. The most important feature in my dataset is ALP. However, the mean of ALP is 219 and the median of ALP is 188 while the actual normal range in reality is 44-147. So as the third important feature D_B. Its mean is 0.39 while the actual normal range in reality is below 0.3. Therefore, it may cause an issue when comparing with true normal range.

## Future direction

For my future direction, first thing I could do is to remove those duplicates in my dataset since it has already affected my prediction.

Second, I can try to obtain real dataset from my former co-workers if possible. Then, I would like to include more features that is correlated with liver disease, such as Gamma-glutamyltransferase (GGT), L-lactate dehydrogenase (LD), Prothrombin time (PT), 5'-nucleotidase, etc.

Last but not least, I would like to try more models since I just used Logistic regression with single and multiple features, decision tree and random forest In this final project. However, there are still plenty of other machine learning methods that I can explore with. 
