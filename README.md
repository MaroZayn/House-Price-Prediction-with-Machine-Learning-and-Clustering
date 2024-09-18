# House-Price-Prediction-with-Machine-Learning-and-Clustering
Overview
This project predicts house prices using a variety of machine learning models and clusters housing data using KMeans to group similar locations. It utilizes various regression techniques, data visualization, and clustering algorithms to enhance prediction accuracy and identify optimal models for different regions.

Features
Machine Learning Models: Comparison of multiple regression models including Linear Regression, Ridge Regression, ElasticNet, Random Forest, XGBoost, and more.
Data Visualization: Visual representations of relationships between features, correlation matrices, and model performance.
Geospatial Visualization: A Folium map is used to represent the best-performing models for different locations based on clustering.
Clustering with KMeans: Apartments are grouped based on location using KMeans, and the best predictive model for each cluster is identified.
Dataset
The dataset used is housing.csv, which includes features such as house prices, location, and proximity to the ocean.

Libraries Used
Pandas: For data manipulation and analysis.
Numpy: For numerical operations.
Seaborn and Matplotlib: For data visualization.
Scikit-learn: For machine learning algorithms, preprocessing, and model evaluation.
XGBoost: For gradient boosting regression.
Folium: For interactive map visualizations.
Machine Learning Models Used
Linear Regression
Ridge Regression
ElasticNet Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Support Vector Machines (SVR)
K-Nearest Neighbors (KNN)
Artificial Neural Network (MLP)
Polynomial Regression (Degree=2)
Clustering
KMeans clustering is used to group houses based on location (longitude and latitude), and the best model for each cluster is identified using absolute error analysis.

Data Visualization
Correlation heatmaps and scatter plots with regression lines to visualize relationships between features and the target variable.
Box plots and histograms to display the distribution of important features.
Geospatial visualizations using Folium to show clusters and the best predictive model for each region.
Installation
Clone the repository:

bash
Copier le code
git clone https://github.com/MaroZayn/house-price-prediction.git
Install the required libraries:

bash
Copier le code
pip install -r requirements.txt
Run the project notebook:

bash
Copier le code
jupyter notebook Projet_House_Prediction.ipynb
How It Works
Data Preprocessing: The data is cleaned, and missing values are handled. Categorical features are encoded and scaled.
Model Training: Various machine learning models are trained on the dataset. The Root Mean Square Error (RMSE) and R-squared scores are used to evaluate model performance.
Clustering: Apartments are clustered using KMeans, and for each cluster, the model that provides the best prediction is identified.
Geospatial Mapping: Predictions and the best model for each cluster are visualized on a map using Folium.
Model Comparison
The models are evaluated and compared based on RMSE and R-squared scores. A bar plot is used to visualize the performance of each model, helping to identify the most accurate model.

Future Enhancements
Improve feature selection and add more features to the dataset.
Tune hyperparameters of the machine learning models for better performance.
Explore additional clustering techniques to improve regional predictions.
