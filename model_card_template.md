# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier that predicts whether a person's income is more than $50,000 per year based on various demographic data. I used the Scikit-Learn library to build this model, and it was trained using default settings, which include 100 decision trees. The model was created for a project that shows how to build a machine learning pipeline and deploy it using FastAPI.

## Intended Use
The main use of this model is to demonstrate how to deploy a machine learning model with FastAPI for educational purposes. It can predict income levels based on input features like age, education, occupation, and race. This model is not meant for real-world decision-making where the outcomes could have serious consequences.

## Training Data
The model was trained using the "Census Income" dataset from the UCI Machine Learning Repository. The dataset contains about 30,000 rows of data, including demographic and employment-related information such as age, work class, education level, marital status, and more. I used one-hot encoding to handle categorical data and scaled numerical features to help the model learn better. The target variable for this model is "salary," which indicates if someone's income is over $50,000 per year.

## Evaluation Data
The model was tested on a subset of the same Census Income dataset, which wasn't used during the training phase. This test dataset has around 6,000 rows of data with the same features as the training dataset. Using this test set, I evaluated how well the model performs on new, unseen data.

## Metrics
I evaluated the model using the following metrics:
- **Precision**: 0.7419
- **Recall**: 0.6384
- **F1 Score**: 0.6863

These metrics help understand the model's performance. Precision tells us how many of the positive predictions were correct, recall shows how well the model found all the actual positive cases, and the F1 Score gives a balance between precision and recall.

## Ethical Considerations
There are some ethical concerns with using this model:
- **Bias**: The model was trained on historical data, which may have biases, such as racial or gender biases, that could affect predictions.
- **Privacy**: The dataset contains sensitive personal data, so it's important to use this model in a way that respects people's privacy.
- **Misuse**: This model shouldn't be used for real-life decisions, especially in areas like hiring, loans, or law enforcement, where fairness and accuracy are crucial.

## Caveats and Recommendations
The model is based on historical data and might not perform well on new data or populations that were not included in the training set. Be careful when using this model, and don't rely on it for decisions that have high stakes. It's a good idea to regularly update and retrain the model with new data to keep it accurate and fair. If you plan to use it more broadly, consider adding methods to reduce bias and improve fairness.