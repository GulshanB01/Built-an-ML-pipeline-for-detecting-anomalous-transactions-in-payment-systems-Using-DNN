# Built-an-ML-pipeline-for-detecting-anomalous-transactions-in-payment-systems-Using-DNN

As part of my journey through a Deep Learning course, I took on one of the most critical challenges in e-commerce and digital payments: detecting fraudulent transactions in highly imbalanced datasets. In Payment App, Pay to seller platforms and reviews — fraud detection plays a pivotal role in ensuring trust and security. Fraudulent activities typically represent <1% of all transactions, making detection both high-impact and technically demanding.

To address this, I:

 Performed Exploratory Data Analysis (EDA): Identified key behavioral features that differentiate genuine and fraudulent transactions. Preprocessed Data: Applied normalization, feature scaling, and resampling techniques to balance class distribution and prepare data for training. Built a Deep Neural Network (DNN): Designed a binary classification model, experimenting with dropout, activation functions, optimizers, and loss functions to improve performance. Focused on Robust Evaluation: Went beyond accuracy to emphasize precision, recall, F1-score, and ROC-AUC, ensuring the model was reliable in detecting rare fraud cases without excessive false alarms. Tackled Imbalanced Data Challenges: Leveraged resampling and regularization strategies to minimize overfitting and improve minority-class detection.

This project was not only about building a neural network but also about understanding how AI safeguards trust in large-scale systems like Amazon’s marketplaces and payment services.  Fraud & Anomaly Detection in Financial Transactions Exploratory Data Analysis (EDA)

Fraud accounted for only 0.13% of transactions → highly imbalanced dataset. Fraud occurred mostly in cashout and transfer transactions. Flagging system was ineffective (99.8% of fraud missed). Fraudulent amounts concentrated between ₹1.3L–₹3.6L. Key Insight → Fraud is rare, high-value, and concentrated in specific transaction types.

 Data Preprocessing

Removed irrelevant high-cardinality features (nameOrig, nameDest). Engineered new features: balance_diff_org, balance_diff_dest. One-hot encoded categorical variables (type). Scaled numerical features using MinMaxScaler.

 Handling Class Imbalance

Applied SMOTE oversampling to balance minority fraud cases. Used class weights to prioritize fraud detection during training.

 Deep Neural Network Architecture

Input: 20 engineered features. Hidden Layers: [128 → 64 → 32] neurons with ReLU, each followed by Dropout for regularization. Output: 1 neuron with sigmoid for binary classification. Loss: Binary Crossentropy | Optimizer: Adam. Early Stopping (patience = 10) with best weight restoration.

 Model Evaluation

Confusion Matrix TN: 593,518 | FP: 41,923 FN: 14,886 | TP: 620,555

Performance Metrics

Fraud Detection (Minority Class): Precision: 93.7% Recall: 97.7% F1-score: 95.6%

Non-Fraud Detection (Majority Class):

Precision: 97.6% Recall: 93.4% F1-score: 95.4% Overall Accuracy: 95.5%

 Key Achievement → Despite extreme imbalance (0.13% fraud), the model achieved high recall on fraud detection with balanced performance across classes.

 Conclusion

This project demonstrates how Deep Learning can effectively address rare-event classification problems in financial transactions. By combining feature engineering, imbalance handling (SMOTE, class weighting), and model regularization, the system achieved high reliability in detecting fraudulent activities while minimizing false positives.

 Requirements

Python 3.x TensorFlow / Keras Scikit-learn Imbalanced-learn Pandas, Numpy Matplotlib / Seaborn
