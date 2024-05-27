

---

## 🩺 Breast Cancer Classification with SVM

### Overview

- **Objective**: Classify breast cancer as malignant or benign using Support Vector Machine (SVM) based on the UCI Breast Cancer Wisconsin dataset.
- **Dataset**: The dataset contains 30 features computed from digitized images of fine needle aspirates (FNAs) of breast masses.

### Steps

1. **📂 Data Loading**
    - **Action**: Load the Breast Cancer dataset from `sklearn.datasets`.
    - **Details**: The dataset includes features such as mean radius, mean texture, mean perimeter, mean area, and mean smoothness of the cell nuclei present in the image.

2. **🔄 Data Preparation**
    - **Action**: Create a DataFrame from the dataset and add a target column.
    - **Details**: Convert the data into a pandas DataFrame for easier manipulation and add a target column to denote whether the tumor is malignant or benign.

3. **🔍 Exploratory Data Analysis (EDA)**
    - **Action**: Perform initial data analysis.
    - **Details**:
      - Check the shape of the dataset to understand its dimensions.
      - Use `info()` to inspect data types and non-null counts to ensure data integrity.
      - Check for missing values to address any data quality issues.

4. **🤖 Model Training**
    - **Action**: Train the Support Vector Machine model.
    - **Details**:
      - Split the data into features (`x`) and target (`y`).
      - Further split the data into training and testing sets (80% training, 20% testing) using `train_test_split`.
      - Initialize the SVM model and train it using the training data.

5. **📈 Model Evaluation**
    - **Action**: Evaluate the model's performance.
    - **Details**:
      - Predict the outcomes for the training set and compute evaluation metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 score.
      - Predict the outcomes for the test set and compute the same evaluation metrics to compare performance.

6. **🌈 Decision Boundary Visualization**
    - **Action**: Visualize the decision boundary of the SVM.
    - **Details**:
      - Select the first two features of the dataset for a two-dimensional plot.
      - Train the SVM model using these two features.
      - Use `DecisionBoundaryDisplay` to plot the decision boundary and overlay the data points on the graph.

7. **📊 ROC Curve Plotting**
    - **Action**: Plot the ROC curve to evaluate model performance.
    - **Details**:
      - Binarize the target labels for multiclass classification.
      - Split the data into training and testing sets using the first two features.
      - Train an SVM model using One-vs-Rest strategy and compute predicted probabilities.
      - Compute the ROC curve and the Area Under the Curve (AUC) score.
      - Plot the ROC curve to visualize the trade-off between the true positive rate and false positive rate.

### Results

- 🖼️ Visualization of the decision boundary for the first two features.
- 📊 Evaluation metrics (MSE, MAE, R2 score) for both training and test sets.
- 📉 ROC curve illustrating the model's performance.

### Dependencies

- 🐍 Python 3.x
- 📦 `scikit-learn`
- 🧮 `pandas`
- 📈 `matplotlib`



#### Advantages

1. **High Accuracy**: SVMs are known for their accuracy in classification tasks, particularly for binary classifications like this breast cancer dataset.
2. **Effective in High Dimensions**: SVMs work well with high-dimensional data, which is beneficial given the 30 features in the breast cancer dataset.
3. **Robust to Overfitting**: With the appropriate choice of kernel and regularization parameters, SVMs can be robust to overfitting, especially in high-dimensional spaces.
4. **Clear Margin of Separation**: SVMs create a clear margin of separation between classes, which can be visualized in lower dimensions.

#### Disadvantages

1. **Computationally Intensive**: Training an SVM can be computationally intensive, especially with large datasets and a complex kernel.
2. **Memory Usage**: SVMs require significant memory, making them less suitable for extremely large datasets.
3. **Parameter Tuning**: SVMs require careful tuning of hyperparameters like the regularization parameter (C) and the kernel parameters, which can be time-consuming.
4. **Not Easily Interpretable**: The decision boundary created by SVMs, especially with non-linear kernels, can be difficult to interpret compared to models like decision trees.

