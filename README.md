🚢 Titanic Survival Prediction with Neural Networks

This project explores the use of Simple Neural Networks and Deep Neural Networks to predict passenger survival on the Titanic using the classic titanic.csv dataset.

📌 Project Overview

The goal of this project is to build and compare different neural network architectures to classify whether a passenger survived or not based on features such as:

Age
Sex
Passenger class (Pclass)
Fare
Number of siblings/spouses aboard
Number of parents/children aboard
Embarked port
🧠 Models Implemented
1. Simple Neural Network
Input layer
One hidden layer
Output layer (binary classification)
2. Deep Neural Network
Multiple hidden layers
Non-linear activations (ReLU)
Improved capacity for feature learning
⚙️ Tech Stack
Python 🐍
NumPy
Pandas
Scikit-learn
TensorFlow / Keras (or PyTorch, depending on your implementation)
Matplotlib / Seaborn (for visualization)
📂 Dataset

The dataset used is titanic.csv, which contains passenger information from the Titanic disaster.

Typical columns include:

Survived (target)
Pclass
Sex
Age
Fare
Embarked
🔧 Data Preprocessing
Handling missing values (Age, Embarked)
Encoding categorical variables (Sex, Embarked)
Feature scaling (Standardization / Normalization)
Train-test split
🚀 How to Run
Clone the repository:
git clone https://github.com/your-username/titanic-neural-network.git
cd titanic-neural-network
Install dependencies:
pip install -r requirements.txt
Run the training script:
python train.py
📊 Evaluation Metrics

Models are evaluated using:

Accuracy
Loss
Confusion Matrix
Precision & Recall
📈 Results
Model	Accuracy
Simple Neural Network	~75-80%
Deep Neural Network	~80-85%

(Results may vary depending on preprocessing and hyperparameters)

📉 Visualization
Training vs Validation Loss
Accuracy curves
Feature distributions
🔍 Key Insights
Feature engineering significantly improves performance
Deep networks capture complex relationships better
Overfitting can occur without proper regularization
📌 Future Improvements
Hyperparameter tuning
Cross-validation
Use of advanced architectures (Dropout, BatchNorm)
Ensemble methods
