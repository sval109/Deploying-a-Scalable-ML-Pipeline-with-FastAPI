import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data



# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    # Trains a machine learning model and returns it.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    # Validates the trained machine learning model using precision, recall, and F1
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1.0, zero_division=1)
    return precision, recall, fbeta



def inference(model, X):
    # Run model inferences and return the predictions.
    preds = model.predict(X)
    return preds


def save_model(model, model_path='model.pkl'):
    # Serializes model to a file.
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path='model.pkl'):
    # Loads pickle file from `path` and returns it."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    # Filter the data for the slice value
    data_slice = data[data[column_name] == slice_value]

    # Process the data slice
    X_slice, y_slice, _, _ = process_data(
        X=data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make predictions on the sliced data
    preds = inference(model, X_slice)

    # Compute and return the metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
