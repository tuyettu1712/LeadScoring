# Lead Scoring Model for X Education

## Problem Statement

X Education, an online education company offering industry-relevant courses, faces a challenge with low lead conversion rates.  Despite attracting a large number of website visitors through various marketing channels, only a small percentage (30 out of 100) convert into paying customers on average. To address this, X Education aims to identify the most promising leads, termed "Hot Leads," and improve their overall lead conversion rate.

The objective is to build a model that can assign a lead score to each prospect based on their demographics, behavior, and preferences. A higher lead score indicates a greater likelihood of conversion, allowing the company to focus their sales and marketing efforts on the most potential customers. The CEO's target is to achieve an 80% lead conversion rate using this model.

## Goals and Objectives

*   Develop a logistic regression model to assign lead scores ranging from 0 to 100 to each lead.
*   Enable X Education to effectively target potential leads by distinguishing "hot" leads (high conversion probability) from "cold" leads (low conversion probability).
*   Create a model that is adaptable to future changes in company requirements and data landscape.

## Process/Approach

This project follows a structured approach to build and evaluate the lead scoring model:

**1. Data Understanding:**

*   **Data Loading and Initial Inspection:** The dataset `Leads.csv` was loaded into a Pandas DataFrame. The initial inspection included checking the data shape (9240 rows, 37 columns), data types using `df.info()`, and descriptive statistics using `df.describe()`.
*   **Key Observations:**
    *   The dataset contains 9240 leads with 37 features.
    *   'Converted' is the target variable, indicating lead conversion (binary - 0 or 1).
    *   Several columns contain 'Select' values, which are identified as default options and treated as missing values.
    *   Missing values are present in various columns, some exceeding 20% and even 70%.

**2. Exploratory Data Analysis (EDA):**

*   **Handling 'Select' Values:**  'Select' values were replaced with `NaN` to properly represent missing data.
*   **Handling Missing Data:**
    *   Columns with more than 70% missing values (`How did you hear about X Education`, `Lead Profile`, `Lead Quality`) were dropped as imputation would be unreliable.
    *   For columns with 20-70% missing values, a column-by-column approach was adopted.
        *   `Lead Quality`: Missing values were imputed with "Unknown" to preserve information, as it is a crucial metric despite missing data.
        *   Asymmetrique Index/Score columns: `Asymmetrique Profile Index`, `Asymmetrique Activity Score`, `Asymmetrique Profile Score` were dropped due to high missing percentage and redundancy with `Asymmetrique Activity Index`. Missing values in `Asymmetrique Activity Index` were imputed with "Unknown".
        *   `City`: Missing values were imputed with 'Mumbai' as it is the most frequent city.
        *   `Specialization`: Missing values were imputed with 'Others' to account for specializations not listed.
        *   `Tags`: Missing values were imputed with 'Unknown' after grouping similar tag categories for better analysis.
        *   `What matters most to you in choosing a course`: Dropped due to extreme imbalance and low variance.
        *   `Country`: Dropped as most data is centered in India and provides little predictive value.
        *   `What is your current occupation`: Missing values imputed with 'Unknown' after regrouping categories for better representation.
    *   Rows with remaining missing values in `TotalVisits`, `Page Views Per Visit`, `Lead Source`, and `Last Activity` (less than 2% missing) were dropped as they are a small portion of the data and imputation might introduce bias.
*   **Outlier Treatment:** Box plots were used to identify outliers in numerical columns (`TotalVisits`, `Page Views Per Visit`). Capping was applied at the 95th percentile for `TotalVisits` and `Page Views Per Visit` to handle extreme outliers without removing data.
*   **Target Variable Analysis ('Converted'):** The distribution of the target variable was checked, revealing a moderate class imbalance (37.9% conversion rate).
*   **Numerical Variable Analysis:** Box plots were used to analyze the relationship between numerical variables (`TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit`) and the target variable 'Converted'. Correlation heatmap was used to check for multicollinearity among numerical variables and their correlation with 'Converted'.
    *   `Total Time Spent on Website` showed a strong positive correlation with conversion.
*   **Categorical Variable Analysis:** Bar plots with overlaid conversion rate lines were used to analyze the relationship between categorical variables and the target variable.  Categorical variables like `Lead Quality`, `Lead Source`, `Last Activity`, and `Specialization` were analyzed in detail to understand their impact on lead conversion.
    *   `Lead Source` and `Last Activity` categories were regrouped to create more meaningful insights due to scattered and granular categories.

**3. Data Preprocessing for Modeling:**

*   **Encoding Binary Variables:** Binary categorical columns with 'Yes/No' values were converted to numerical (1/0) format.
*   **Encoding Categorical Variables:**
    *   `Asymmetrique Activity Index`:  Numerical mapping was applied to convert categories ('High', 'Medium', 'Low', 'Unknown') to numerical scores (1, 2, 3, 4).
    *   Remaining categorical columns (`Lead Origin`, `Lead Source`, `Last Activity`, `Specialization`, `What is your current occupation`, `Tags`, `Lead Quality`, `City`, `Last Notable Activity`) were converted to dummy variables using one-hot encoding with `pd.get_dummies()`.
*   **Splitting Data:** The dataset was split into training (80%) and testing (20%) sets using `train_test_split`.
*   **Feature Scaling:** Min-Max scaling was applied to numerical features (`TotalVisits`, `Page Views Per Visit`, `Total Time Spent on Website`) in both training and testing sets to ensure features are on a similar scale.
*   **Indexing:** 'Lead Number' was used as index for `y_train` and `y_test` and then dropped from feature sets. Indexing was reset for `X_train` and `X_test`.

**4. Model Building (Logistic Regression):**

*   **Feature Selection using RFE:** Recursive Feature Elimination (RFE) with Logistic Regression was used to select the top 15 most significant features for the model.
*   **Model 1 - Initial Logistic Regression:** A logistic regression model (`model1`) was built using the features selected by RFE.
*   **Variance Inflation Factor (VIF) Check:** VIF was calculated for the features in `model1` to detect multicollinearity. 'Tags_Unknown' was found to have a high VIF (6.09).
*   **Model 2 - Refined Logistic Regression:** 'Tags_Unknown' was dropped to reduce multicollinearity, and a second logistic regression model (`model2`) was built using the remaining features.

**5. Model Evaluation:**

*   **Model Summary and Metrics:** `model2` summary was analyzed to assess feature significance (p-values) and model fit.
*   **ROC Curve and AUC:** ROC curve was plotted and AUC score calculated (AUC = 0.91 for training data), indicating excellent model discrimination.
*   **Optimal Cut-off Determination:**
    *   Probability cut-off analysis: Accuracy, sensitivity, and specificity were calculated for various probability cut-offs (0.0 to 0.9).
    *   Precision-Recall curve: Precision and recall trade-off was visualized to find a balanced cut-off.
    *   A cut-off of 0.4 was chosen as it provided a good balance between sensitivity and specificity, aiming to maximize true positives while controlling false positives.
*   **Confusion Matrix and Classification Metrics:** Confusion matrix, accuracy, sensitivity, specificity, precision, FPR, negative predictive value, and F1-score were calculated for both the training and test sets at the 0.4 cut-off.

## Model Evolution

The model building process involved iterative refinement:

*   **Initial Feature Selection (RFE):** RFE helped to reduce the dimensionality of the feature space and identify the most predictive features for lead conversion. 15 features were initially selected based on RFE ranking.
*   **Multicollinearity Handling (VIF):** VIF analysis revealed multicollinearity issues, particularly with 'Tags_Unknown'. Dropping this feature improved model stability and interpretability.
*   **Cut-off Optimization:** Instead of using the default 0.5 cut-off, a more optimal cut-off of 0.4 was determined through probability cut-off analysis and precision-recall curve visualization. This improved the balance between sensitivity and specificity, making the model more practical for business use.

**Selected Features in Final Model (model2):**

*   Do Not Email
*   Total Time Spent on Website
*   Lead Origin_Lead Add Form
*   Lead Source_Partner Sites
*   What is your current occupation_Unknown
*   What is your current occupation_Working
*   Tags_Contact_Issue
*   Tags_Current_Student
*   Tags_Lost_Closed
*   Tags_Not_Interested
*   Tags_Others
*   Lead Quality_Not Sure
*   Lead Quality_Unknown
*   Lead Quality_Worst

## Conclusion

The final logistic regression model  demonstrates strong performance in predicting lead conversion for X Education.


**Key Insights:**

*   The model achieves a high AUC of 0.91 on the training data, indicating excellent discriminatory power.
*   **'Total Time Spent on Website'** is a highly significant positive predictor of lead conversion, highlighting the importance of website engagement.
*   **'Lead Quality'** and **'Tags'** categories are crucial features, indicating the importance of lead profiling and behavior tracking.
*   **'Lead Origin_Lead Add Form'** and **'Lead Source_Partner Sites'** are positive predictors, suggesting the effectiveness of lead acquisition methods through lead forms and partner networks.

**Business Value for X Education:**

By implementing this lead scoring model, X Education can:

*   **Prioritize Hot Leads:** Focus sales and marketing efforts on leads with higher scores, increasing conversion efficiency.
*   **Improve Conversion Rate:** Achieve a significantly higher conversion rate than the current 30% by targeting potential customers more effectively, potentially reaching or exceeding the CEO's target of 80% (though direct 80% conversion rate is a target, this model significantly improves targeting).
*   **Optimize Marketing Spend:** Allocate marketing resources more efficiently by focusing on channels and lead characteristics that contribute to higher conversion scores.
*   **Personalize Customer Engagement:** Tailor communication and engagement strategies based on lead scores and associated characteristics.

This model provides a valuable tool for X Education to enhance its lead conversion process and drive business growth.
