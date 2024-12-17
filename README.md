

---

# Supermarket Sales Insights: 📊 KPI Analysis & Prediction Models

## Overview
This project aims to provide a comprehensive analysis of supermarket sales data to gain insights and make predictions. The dataset includes details such as sales IDs, branch and city information, customer types, product details, unit prices, quantities sold, sales tax, and total price after tax. Using various machine learning (ML) algorithms, this project performs exploratory data analysis (EDA), feature engineering, and predictive modeling to optimize sales insights.

## Project Highlights
- **Explores sales trends and customer behavior** across major cities (New York, Chicago, Los Angeles).
- **Analyzes KPIs** like total sales, customer demographics, reward distribution, and more.
- **Uses ML algorithms** including Linear Regression, Random Forest, Gradient Boosting (XGBoost), Decision Trees, and SVR for predictive modeling.
- **Incorporates feature engineering and data preprocessing** steps such as handling missing values, encoding categorical data, and normalizing numerical features.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Linear Regression, Random Forest, Gradient Boosting (XGBoost), Decision Trees, SVR

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/supermarket-sales-insights.git
   cd supermarket-sales-insights
   ```

2. **Setup environment**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Exploratory Data Analysis (EDA)**:
   - Load the dataset and perform basic data cleaning.
   - Visualize the distribution of key features like sales, customer types, and product categories using histograms and bar plots.
   - Detect and handle outliers with box plots.
   - Correlation heatmap to understand relationships between features.

2. **Feature Engineering**:
   - Handle missing values for categorical and numerical columns.
   - Encode categorical features using label encoding.
   - Normalize numerical features for better model performance.

3. **Model Building**:
   - **Linear Regression**:
     ```python
     from sklearn.linear_model import LinearRegression
     # Train the model
     model_lr = LinearRegression()
     model_lr.fit(X_train, y_train)
     # Predictions and evaluation
     predictions_lr = model_lr.predict(X_test)
     ```
     - **Math**: 
       \[
       \hat{y} = \beta_0 + \beta_1 \times X
       \]
       Here, \(\hat{y}\) is the predicted value, \(\beta_0\) is the intercept, \(\beta_1\) are the coefficients, and \(X\) represents the features.

   - **Random Forest Regressor**:
     ```python
     from sklearn.ensemble import RandomForestRegressor
     model_rf = RandomForestRegressor()
     model_rf.fit(X_train, y_train)
     predictions_rf = model_rf.predict(X_test)
     ```
     - **Math**:
       - Decision Trees with bagging (ensemble method) enhance accuracy and handle complex relationships.

   - **Gradient Boosting (XGBoost)**:
     ```python
     from xgboost import XGBRegressor
     model_xgb = XGBRegressor()
     model_xgb.fit(X_train, y_train)
     predictions_xgb = model_xgb.predict(X_test)
     ```
     - **Math**:
       - XGBoost uses gradient boosting on decision trees to optimize the loss function iteratively, enhancing predictive performance.

   - **Decision Trees**:
     ```python
     from sklearn.tree import DecisionTreeRegressor
     model_dt = DecisionTreeRegressor()
     model_dt.fit(X_train, y_train)
     predictions_dt = model_dt.predict(X_test)
     ```
     - **Math**:
       - Divides the dataset into homogenous subsets based on the best feature splits.

   - **SVR (Support Vector Regression)**:
     ```python
     from sklearn.svm import SVR
     model_svr = SVR(kernel='rbf')
     model_svr.fit(X_train, y_train)
     predictions_svr = model_svr.predict(X_test)
     ```
     - **Math**:
       - Finds the best-fitting hyperplane that predicts the target variable.

4. **Evaluation**:
   - **Metrics**: 
     - **Mean Squared Error (MSE)**: Measures average squared differences between predicted and actual values.
     - **R-squared**: Indicates goodness of fit, explaining the proportion of variance.
     - **Mean Absolute Error (MAE)**: Measures the average error between predictions and actual values.
   - **Comparison**:
     - Compare the performance of each model using these metrics to determine which provides the best accuracy for the dataset.

5. **Feature Importance**:
   - Visualize the most important features using bar plots for decision tree-based models (Random Forest, XGBoost).
   - **Code**:
     ```python
     feature_importance = model_rf.feature_importances_
     sorted_idx = np.argsort(feature_importance)
     plt.figure(figsize=(10, 6))
     plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx])
     plt.title('Feature Importance')
     plt.show()
     ```

## Conclusion
This project demonstrates the application of machine learning techniques to analyze and predict sales data. By using various regression models and evaluating their performance, we can better understand sales trends and optimize business decisions. The process involves data cleaning, feature engineering, model training, and evaluation, providing a holistic approach to data science.

## Future Work
- Explore additional features such as time series data or customer feedback.
- Implement advanced deep learning models like neural networks for more complex predictive tasks.
- Optimize hyperparameters for better model performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- The dataset was provided for practice purposes and does not reflect actual sales data.
- Thanks to the contributors for their help and feedback during the project.

---

