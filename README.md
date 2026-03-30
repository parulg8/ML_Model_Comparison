# ML Model Comparison (Iris Dataset)

## Objective
The objective of this assignment is to compare the performance of three machine learning algorithms:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree

The goal is to evaluate how different models perform on the same dataset and understand their behavior.

---

## Dataset
The Iris dataset is a well-known dataset used for classification tasks. 

It contains:
* 150 samples of iris flowers
* 3 classes (Setosa, Versicolor, Virginica)
* 4 features: sepal length, sepal width, petal length, petal width

This dataset is clean, balanced, and ideal for beginners.

---

## Approach
The following steps were followed:
1. Loaded the Iris dataset using scikit-learn
2. Split the dataset into training (80%) and testing (20%) sets
3. Applied feature scaling using StandardScaler
4. Trained three models:
   * Logistic Regression
   * K-Nearest Neighbors (KNN)
   * Decision Tree
5. Made predictions on the test dataset
6. Evaluated model performance using accuracy
7. Compared results in a tabular format

---

## Steps Performed
* Data loading and preprocessing
* Train-test split
* Feature scaling
* Model training
* Prediction
* Evaluation using accuracy and confusion matrix

---

## Results
| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 1.00     |
| KNN                 | 1.00     |
| Decision Tree       | 1.00     |

All models achieved perfect accuracy on the test dataset.

---

## Best Model and Explanation
All three models achieved 100% accuracy on the Iris dataset.
This is because the dataset is small, clean, and well-separated, making it easy for different algorithms to classify the data correctly.

However, the models differ in how they work:
* Logistic Regression assumes linear relationships between features
* KNN uses distance between data points to classify
* Decision Tree creates rule-based splits

Even though accuracy is the same, KNN is often preferred for this dataset because it works very well with small and well-structured data.
In more complex datasets, these models would likely perform differently.

---

## Difficulties Faced
* Interpreting the results when all models achieved the same accuracy (1.0), making it difficult to clearly identify the “best” model
* Understanding when and why feature scaling is necessary, especially since not all models require it
* Grasping the importance of train-test splitting and how it impacts model evaluation
* Comparing fundamentally different algorithms (distance-based, linear, and tree-based) in a meaningful way

---

## Resolutions
* Analyzed the nature of the dataset and concluded that identical accuracy was due to its simplicity and clear class separation
* Studied how different models work to understand why they can still perform equally well despite different approaches
* Learned that feature scaling is especially important for distance-based models like KNN, even if others are less affected
* Reinforced the concept of train-test split to ensure proper evaluation on unseen data
* Used additional evaluation methods like confusion matrix to gain deeper insight into model performance


---

## Learnings
* How to train and evaluate multiple ML models
* Importance of splitting data into training and testing sets
* Role of feature scaling in machine learning
* Differences between Logistic Regression, KNN, and Decision Tree
* Why model performance can vary depending on the dataset

---

## Conclusion
All three models performed equally well on the Iris dataset due to its simplicity and clear class separation.

This experiment shows that model performance depends heavily on the dataset. While all models performed perfectly here, more complex datasets would highlight differences in their effectiveness.

---

## How to Run This Project

### Prerequisites
* IDE installed
* Import required libraries:

  ```
  pip install pandas numpy scikit-learn
  ```

### Steps to Run
1. Clone the repository:

   ```
   git clone <your-repo-link>
   ```
2. Navigate to the project folder:

   ```
   cd ml-model-comparison
   ```
3. Run the script:

   ```
   python main.py
   ```

---

## Author
Parul Ghosh
