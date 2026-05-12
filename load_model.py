import joblib
import os
from pathlib import Path

# Load the trained model
script_dir = Path(__file__).resolve().parent
model_path = script_dir / "loan_model.pkl"

if model_path.exists():
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    print(f"\nModel Type: {type(model).__name__}")
    print(f"Model Parameters: {model.get_params()}")
    print(f"\nNumber of trees: {model.n_estimators}")
    print(f"Number of features: {model.n_features_in_}")
    print(f"\nFeature Importances:")
    for i, importance in enumerate(model.feature_importances_):
        print(f"  Feature {i}: {importance:.4f}")
else:
    print(f"Model file not found at {model_path}")
