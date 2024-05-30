# ML-Poli-Sci


Slides: https://docs.google.com/presentation/d/1Qzgl6P9cNUejpLUx-ivtGDouEj2i82tKl3jF-sBwdbU/edit?usp=sharing

Missing value imputation-R: https://github.com/IQSS/amelia & https://gking.harvard.edu/amelia

Missing value imputation-sklearn:https://scikit-learn.org/stable/modules/impute.html

## log

- 2024/4/18-5/30:
   - add variables: VCF0006a” or “Unique Respondent Number (Cross-year ID for panel cases), voter/non-voter, vote_D/vote_R to the state-wise prediction results:
        - VCF0006a: added
        - voter/non-voter,vote_D/vote_R: already in the data

   - add the one model trained on the whole data and the state-wise model prediction results:
        - to do in Friday

   - add the citation of the ML term in documents
   - clean the code, make it:
        - script-run for new data test (apply curent model to new data)
        - script-run for new data training (model update)
        - visulization and store results script


- 2024/4/11:
   - send documents on lgistic regression + elastic net to the professor
   - make the documetns on data process ( one-hot  ) and missing value imputation
   - build a table to show the gap between the "vote-D" and "vote-R" for whole data and state-wise data


- 2024/3/26:
    - model did not work well on "intend-vote" group due to the imbalance of the data
    - try some imbalanced data handling methods, like SMOTE, ADASYN, and RandomOverSampler, but did not work well
    - tried some advanced/non-linear models, like GBTtree, RBF-SVM, and ensemble models, like AdaBoost , but did not work well
    - clean the code and add some comments


- 2024/3/12:
    - save and finished almost all stat-based analysis
    - start to build the feature-importance model (Log-Reg)
    - focus on the "WA" state group


- 2024/2/23:
 - add table for "intend to vote" but final "non-voter" for the white/black group in different area
 - add year-based plotting for the changing ratio

- get some  hypotheses to verify(based on "urban-rural"-feature, not miss out!): 

    - Blacks in urban America are more likely to vote and vote for the Democratic candidates than are Blacks in rural America.
 
    - Blacks in suburban America are more likely to vote for the Republican candidates than are Blacks in urban America.
 
    - Whites in rural America are more likely to vote Republicans than are Whites in urban or suburban America.
 
    - White non-voters are more likely to live in rural American than in urban America.

- state-based analysis: focus on WA

- foucus on final non-voters who intend to vote

- after feature filter-out, check the performance with only top-5/10/20 features 




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

