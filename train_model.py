import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# load the census.csv data
project_path = "/mnt/c/Users/stacy/PycharmProjects/pythonProject9.12/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
    )

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")

save_model(model, model_path=model_path)
print(f"Model saved to {model_path}")

with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)
print(f"Model saved to {encoder_path}")

# load the model
print(f"Loading model from {model_path}")
model = load_model(model_path)

# load the encoder
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    for slice_value in sorted(test[col].unique()):
        data_slice = test[test[col] == slice_value]
        count = len(data_slice)

        # Skip slices with insufficient data
        if count < 5:  # Choose a threshold based on your context
            continue

        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slice_value,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)