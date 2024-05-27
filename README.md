
---

## 🩺 Breast Cancer Classification using Support Vector Machine (SVM)

### Overview

- **Objective**: Classify breast cancer as malignant or benign using Support Vector Machine (SVM) based on the UCI Breast Cancer Wisconsin dataset.
- **Dataset**: The dataset contains 30 features computed from digitized images of fine needle aspirates (FNAs) of breast masses.

### Steps

1. **📂 Data Loading**
    - Load the Breast Cancer dataset from `sklearn.datasets`.

2. **🔄 Data Preparation**
    - Create a DataFrame from the dataset.
    - Add a target column to the DataFrame indicating the class labels (malignant or benign).

3. **🔍 Exploratory Data Analysis (EDA)**
    - Check the shape of the dataset.
    - Inspect dataset information and data types.
    - Verify if there are any missing values.

4. **🤖 Model Training**
    - Split the dataset into features (`x`) and target (`y`).
    - Further split the data into training and testing sets (80% training, 20% testing).
    - Train an SVM model with the training data.

5. **📈 Model Evaluation**
    - Predict the training set results.
    - Calculate and print Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 score for the training set.
    - Predict the test set results.
    - Calculate and print MSE, MAE, and R2 score for the test set.

6. **🌈 Decision Boundary Visualization**
    - Select the first two features for visualization.
    - Train an SVM model using the selected features.
    - Plot the decision boundary using `DecisionBoundaryDisplay`.
    - Create a scatter plot of the data points.

7. **📊 ROC Curve Plotting**
    - Binarize the labels for multiclass classification.
    - Split the data into training and testing sets using the first two features.
    - Train an SVM model using One-vs-Rest strategy.
    - Predict probabilities and compute the ROC curve.
    - Plot the ROC curve and calculate the area under the curve (AUC).

### Results

- 🖼️ Visualization of the decision boundary for the first two features.
- 📊 Evaluation metrics (MSE, MAE, R2 score) for both training and test sets.
- 📉 ROC curve illustrating the model's performance.

### Dependencies

- 🐍 Python 3.x
- 📦 `scikit-learn`
- 🧮 `pandas`
- 📈 `matplotlib`

### Usage

1. **📥 Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-svm.git
   cd breast-cancer-svm
   ```
