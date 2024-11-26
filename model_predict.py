import pandas as pd
import joblib

# Function to load model7 on demand
def load_model7():
    model7_path = 'C:\\Users\\tanwa\\OneDrive\\Desktop\\Genetic\\gen\\models\\model7.pkl'
    model7 = joblib.load(model7_path)
    return model7

# Function to load model15 on demand
def load_model15():
    model15_path = 'C:\\Users\\tanwa\\OneDrive\\Desktop\\Genetic\\gen\\models\\model15.pkl'
    model15 = joblib.load(model15_path)
    return model15

def predict_genetic_disorder(input_data):
    # Load models on demand
    model7 = load_model7()
    model15 = load_model15()

    # Convert the input JSON data to a pandas DataFrame
    single_input = pd.DataFrame(input_data)

    # Columns from training dataset
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
        'Heart Rate (rates/min)',
        'Respiratory Rate (breaths/min)',
        'Folic acid details (peri-conceptional)',
        'History of anomalies in previous pregnancies',
        'Autopsy shows birth defect (if applicable)',
        'Assisted conception IVF/ART',
        'Symptom 4',
        'Follow-up',
        'Birth defects'
    ]

    # Adjust the input to match expected columns
    single_input = single_input.reindex(columns=expected_columns, fill_value=0)

    # Make predictions
    final_pred1 = model7.predict(single_input)
    final_pred2 = model15.predict(single_input)

    # Create a DataFrame for submission
    submission = pd.DataFrame()
    submission['Patient Id'] = [1]  # You can modify this if needed
    submission['Genetic Disorder'] = final_pred1
    submission['Disorder Subclass'] = final_pred2

    # Replace numerical values with descriptive strings
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

# Example usage:
input_data = {
    'White Blood cell count (thousand per microliter)': [0.6529786],
    'Blood cell count (mcL)': [-0.5118449],
    'Patient Age': [-0.714285714],
    'Father\'s age': [0],
    'Mother\'s age': [0],
    'No. of previous abortion': [-0.666666667],
    'Blood test result': [2],
    'Gender': [0],
    'Birth asphyxia': [0],
    'Symptom 5': [1],
    'Heart Rate (rates/min)': [0],
    'Respiratory Rate (breaths/min)': [0],
    'Folic acid details (peri-conceptional)': [1],
    'History of anomalies in previous pregnancies': [1],
    'Autopsy shows birth defect (if applicable)': [0],
    'Assisted conception IVF/ART': [1],
    'Symptom 4': [0],
    'Follow-up': [1],
    'Birth defects': [0],
}

# Call the function and get the JSON output
#output_json = predict_genetic_disorder(input_data)

# Print the output
#print(output_json)
