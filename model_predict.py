import os
import gdown
import pandas as pd
import joblib

# Define paths and URLs for models
MODEL_DIR = "models"  # Directory to store models
MODEL1_URL = "https://drive.google.com/uc?id=1cATLRCX35rOPEo5DlrhB8QKZjjcrf5JC"
MODEL2_URL = "https://drive.google.com/uc?id=186kGZFhB1rSPLqqVyKEgO-JcSH0mqlCw"
MODEL1_PATH = os.path.join(MODEL_DIR, "model1.pkl")
MODEL2_PATH = os.path.join(MODEL_DIR, "model2.pkl")

def download_model(model_url, model_path):
    """Download the model file if it doesn't already exist."""
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url} to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(model_url, model_path, quiet=False)

def init_models():
    """Download models at app startup."""
    download_model(MODEL1_URL, MODEL1_PATH)
    download_model(MODEL2_URL, MODEL2_PATH)

def load_models():
    """Load models into memory."""
    print("Loading models into memory...")
    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)
    return model1, model2

# Initialize models at app startup
init_models()
model1, model2 = load_models()  # Models are loaded into memory at startup

def predict_genetic_disorder(input_data):
    """Predict genetic disorder using preloaded models."""
    # Convert the input JSON data to a pandas DataFrame
    single_input = pd.DataFrame(input_data)

    # Expected columns from the training dataset
    expected_columns = [
        'White Blood cell count (thousand per microliter)',
        'Blood cell count (mcL)',
        'Patient Age',
        'Father\'s age',
        'Mother\'s age',
        'No. of previous abortion',
        'Blood test result',
        'Gender',
        'Birth asphyxia',
        'Symptom 5',
        'Heart Rate (rates/min',
        'Respiratory Rate (breaths/min)',
        'Folic acid details (peri-conceptional)',
        'History of anomalies in previous pregnancies',
        'Autopsy shows birth defect (if applicable)',
        'Assisted conception IVF/ART',
        'Symptom 4',
        'Follow-up',
        'Birth defects'
    ]

    # Adjust input to match expected columns
    single_input = single_input.reindex(columns=expected_columns, fill_value=0)

    # Make predictions using preloaded models
    final_pred1 = model1.predict(single_input)
    final_pred2 = model2.predict(single_input)

    # Create a DataFrame for predictions
    submission = pd.DataFrame()
    submission['Patient Id'] = [1]  # Replace with dynamic IDs if needed
    submission['Genetic Disorder'] = final_pred1
    submission['Disorder Subclass'] = final_pred2

    # Replace numerical predictions with descriptive strings
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(0, 'Mitochondrial genetic inheritance disorders')
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(2, 'Single-gene inheritance diseases')
    submission['Genetic Disorder'] = submission['Genetic Disorder'].replace(1, 'Multifactorial genetic inheritance disorders')

    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(0, "Alzheimer's")
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(1, 'Cancer')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(2, 'Cystic fibrosis')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(3, 'Diabetes')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(4, 'Hemochromatosis')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(5, "Leber's hereditary optic neuropathy")
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(6, 'Leigh syndrome')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(7, 'Mitochondrial myopathy')
    submission['Disorder Subclass'] = submission['Disorder Subclass'].replace(8, 'Tay-Sachs')

    # Convert to JSON and return the result
    json_output = submission.to_json(orient='records', lines=True)
    return json_output
