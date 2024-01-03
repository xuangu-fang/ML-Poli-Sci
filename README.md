# ML-Poli-Sci


Slides: https://docs.google.com/presentation/d/1Qzgl6P9cNUejpLUx-ivtGDouEj2i82tKl3jF-sBwdbU/edit?usp=sharing

Missing value imputation-R: https://github.com/IQSS/amelia & https://gking.harvard.edu/amelia

Missing value imputation-sklearn:https://scikit-learn.org/stable/modules/impute.html

## log

2024 Jan 3rd: 

- finish the feature filtering: set the missing-ratio  in recent 20 years as the threshold, and remove the features with missing ratio larger than 0.3-done

- build the simple classifier: logistic regression, random forest, and gradient boosting-doing

### to-do:
- using the small dataset (no missing values) to build the simple classifier
- send email to the professor to ask about the missing values and categorical features
- using the sklearn/R based imputation method to deal with the missing values 


### problem: 
how to deal with the missing data?

- just drop? -> only 10% of the data left( 6000~7000 samples)
- fill with mean? -> too much categorical data
- use the `amelia` package? -> can it deal with the categorical data? -> to check
- use the `sklearn.impute` package? -> can it deal with the categorical data? -> to check
- use 

how to deal with the categorical features?
- should we use the label "UK/don't want answer" as a category, or just drop it/as a missing values?
-  is so, can we use the `sklearn.preprocessing.OneHotEncoder` to encode the categorical features? -> to check

2023 Dec: 

- finish the data collection and process, set the target - Done

