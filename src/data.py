"""Load and split the diabetes dataset.

Uses the unscaled version of the scikit-learn diabetes data so the API takes
real measurements such as age in years and blood pressure in mm Hg.
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Order the model expects. The names on the right are the friendlier
# names used in the API request body.
FEATURES = {
    "age": "age",
    "sex": "sex",
    "bmi": "bmi",
    "bp": "blood_pressure",
    "s1": "total_cholesterol",
    "s2": "ldl",
    "s3": "hdl",
    "s4": "cholesterol_hdl_ratio",
    "s5": "log_triglycerides",
    "s6": "blood_sugar",
}


def load_data():
    """Return the feature matrix X and the disease progression target y."""
    data = load_diabetes(scaled=False)
    return data.data, data.target


def split_data(X, y):
    """Split into 80 percent training and 20 percent test data."""
    return train_test_split(X, y, test_size=0.2, random_state=42)
