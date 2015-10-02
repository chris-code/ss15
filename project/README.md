# PythonProject What's Cooking

The idea of this project is to be able to determine the cuisine of a list of various ingredients with a high credibility. This project is created during the [Python seminar](http://www.ini.rub.de/courses/69-Scientific%20Computing%20with%20Python) held at the [Ruhr-University Bochum](http://www.rub.de).

---

This project consists of the following modules:
- **svm.py:** This file calls the other modules to execute the actual classification tasks. It returns a file to be uploaded to [kaggle](http://www.kaggle.com).
- **readFile.py:** A module used for all file operations, most importantly reading Training and Test data. It also splits documents for cross validation and vectorizes the data using a tf-idf vectorizer.
- **testClassifiers.py:** This module tests different classifiers and returns their mean cross-validation scores. Currently it tests a Random Forest, Extra Trees and a Support Vector Machine. Other tests using AdaBoost and a Bagging classifier have been executed, but have been found lacking for the task at hand and thus have not been implemented in the final project.
- **testCases.py:** The unit test of this project assures the existence and correct format of the training and test files.
- **dataframe_conversion:** This module reads the original training and test files and provides different ways to normalize the ingredients. It outputs the results as csv files.
- **IngredientManipulation:** This module is called by dataframe_conversion.py and provides a tokenization procedure as well as three different approaches to ingredient normalization: A naive approach that removes all tokens it guesses to be adjectives, a way based on NLTK that uses tagging to identify superfluous tokens and a simple concatenation that simply concatenates the multiple token ingredients into a single token.

---

So far the project has produced good results. The svm running on a linear kernel with a penalty parameter of 1.0 has produced the best result so far, classifying 78,59% of the test data correctly and earning a spot in the top 50 of the [What's Cooking leaderboard](https://www.kaggle.com/c/whats-cooking/leaderboard) as of 02/10/2015.
Further tweaking has not yet led to improvement; neither changing the penalty parameter nor preprocessing the ingredients has increased the accuracy and, indeed, caused some misclassifications.
We assume the ExtraTrees classifier to be able to potentially result in a higher accuracy; however, due to the increase in estimators needed to test that theory it would be infeasible to run this classification due to the limited computing power of the provided hardware.