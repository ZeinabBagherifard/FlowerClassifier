# Beginner ML Project: Iris Classification

This project demonstrates a simple machine learning classification task using the famous Iris dataset. It uses scikit-learn to build a Random Forest classifier that predicts the species of iris flowers based on their measurements.

## Dataset

The Iris dataset contains 150 samples of iris flowers, each with 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

And 3 classes:
- Setosa
- Versicolor
- Virginica

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

Run the main script:
```
python main.py
```

This will train the model and print the accuracy on the test set.

## Next Steps

- Experiment with different classifiers (SVM, KNN, etc.)
- Tune hyperparameters
- Visualize the data using matplotlib
- Try other datasets