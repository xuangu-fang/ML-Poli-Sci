# Document on Feature Processing

## Part 1: Selection of Candidate Features

In the initial phase of our feature processing workflow, the selection of candidate features is critically important. This selection is primarily based on expert judgment and domain knowledge, leveraging insights from various specialized fields to create a robust and relevant feature set for our analysis. Here’s a detailed breakdown of our approach:

- **Expert and Domain Knowledge Integration**: Based on Baodong's book and suggestions, we identified variables that are most likely to influence the outcome of interest—political voting behavior. This systematic selection ensures that the features are not only statistically significant but also contextually relevant.

- **Diverse Domains**: The features were selected from eight distinct domains, each contributing unique aspects to the overall analysis. These domains include:
  1. **Contextual Domain**: Features that provide background and situational context to the voting scenarios.
  2. **Identity Domain**: Variables that describe demographic and identity-related aspects of individuals.
  3. **Presidential Politics Domain**: Features specifically related to opinions and engagements concerning presidential campaigns.
  4. **Affect Domain**: Variables that capture emotional and psychological states influencing voting behavior.
  5. **Electoral Engagement Domain**: Features related to the participants' engagement in the electoral process.
  6. **Cognitive Domain**: This includes cognitive factors that could influence decision-making processes.
  7. **Political Inclinations Domain**: Variables that describe the political leanings and inclinations of individuals.
  8. **Socio-Economic Domain (Socio-Eco Domain)**: Features that represent the socio-economic status and conditions of the individuals.

- **Types of Features**:
  - **Numerical Features**: A total of 16 numerical features were selected, providing quantitative data that can be directly analyzed and modeled.
  - **Categorical Features**: We identified 99 categorical features that describe qualitative attributes of the data, which are essential for a nuanced understanding of voter behavior.

This comprehensive selection of 115 variables from varied domains ensures that our feature set is not only multidimensional but also deeply integrated with domain-specific knowledge, providing a strong foundation for subsequent analytical processes.

## Part 2: Feature Processing and Imputation of Missing Values

The primary motivation for this phase of our work is to address the challenges posed by extensive missing values in our dataset, a common issue in political science research that can significantly impact modeling accuracy and insights. We devised a comprehensive approach to mitigate these effects and enhance the quality of our data for modeling.

##### Step 1: Assessing Missing Values
The first step involves a detailed analysis of the missing values across all candidate features. We compute the cumulative missing ratio over multiple decade-long spans to understand the persistence and pattern of missing data across different time periods. For instance, we calculate missing ratios for the last 10 years (2012-2022), 20 years (2002-2022), and so forth. This assessment helps in identifying trends and periods with significant data loss, which are critical for subsequent processing steps. The detailed missing value analysis result is available at: (https://github.com/xuangu-fang/ML-Poli-Sci/blob/master/data/missing_value_analysis.csv)

##### Step 2: Feature Selection Based on Missing Value Thresholds
To ensure the robustness of our model, we adopt criteria that prioritize features with fewer missing values in more recent years, as these are likely to provide more reliable and relevant information. We establish a set of thresholds for missing ratios over different time spans:

- Less than 20% missing in the last 10 years
- Less than 30% missing in the last 20 years
- Less than 40% missing in the last 30 years
- Less than 50% missing in the last 40 years

Based on these criteria, we select 12 numerical features and 52 categorical features for modeling. The detailed list of these features is available at [here](https://github.com/xuangu-fang/ML-Poli-Sci/blob/master/result/state-wise-universal_predict-add-year/_threshold_10_0.2_threshold_20_0.3_threshold_30_0.4_threshold_40_0.5/used_features.txt).

##### Step 3: Processing Features
We process the selected features to ensure consistency and comparability across the dataset:

- **Standardization (Numerical Features)**: We apply standardization to numerical features [1], which involves scaling the data so that it has a mean of zero and a standard deviation of one. This process is essential because it neutralizes the scale of different features, allowing for a fair comparison and combination within models. It also enhances the stability and performance of many machine learning algorithms that are sensitive to the scale of input data.

- **One-Hot Encoding (Categorical Features)**: For categorical features, we convert them into a set of binary features through one-hot encoding[2]. This method transforms each categorical level into a new binary variable, representing the presence (1) or absence (0) of each category. This encoding is crucial for handling non-numeric data, enabling it to be effectively included in mathematical models and ensuring that the algorithm treats each category as a separate entity without any ordinal relationship.

##### Step 4: Imputation of Missing Values
The final step involves handling the remaining missing values in our dataset tailored to the type of processing applied to each feature:

- **Numerical Features**: We impute missing values in standardized features using the median of each feature. The choice of the median is driven by its robustness to outliers, making it a reliable measure for central tendency in skewed distributions.

- **Categorical Features**: For categorical features, we treat missing values as a new category. This approach acknowledges the possibility that missingness itself might be informative and ensures that no data is discarded. After categorization, we apply one-hot encoding to these newly created classes as well.

Through these systematic steps, we effectively manage the presence of missing values while preserving and enhancing the integrity and usability of our data for robust political science modeling.


Reference:

[1]: Zheng, Alice, and Amanda Casari. Feature engineering for machine learning: principles and techniques for data scientists. " O'Reilly Media, Inc.", 2018.

[2]: Seger, Cedric. "An investigation of categorical variable encoding techniques in machine learning: binary versus one-hot and feature hashing." (2018).

