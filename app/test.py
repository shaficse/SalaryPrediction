import joblib
import numpy as np

# Load the model and scaler from files
loaded_model = joblib.load('best_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Take user input for Years of Experience
years_of_experience = 34

# Assuming other features remain constant
job_title_order = 53
education_level = 1
type_title = 4

# Create a NumPy array with the input features
input_features = np.array([[years_of_experience, job_title_order, education_level, type_title]])

# Scale only the "Years of Experience" feature using the loaded scaler
input_features[:, 0] = loaded_scaler.transform(input_features[:, 0].reshape(-1, 1)).flatten()

def test_input():
        try:
            _ = loaded_model.predict(input_feature)
        except Exception as e:
            assert False, f"Model failed to take expected input: {e}"
    
def test_output_shape():    
    try:
        prediction = loaded_model.predict(input_feature)
        assert prediction.shape == (1,), "Output shape is not as expected"
    except Exception as e:
        assert False, f"Output shape test failed: {e}"

if __name__ == '__main__':
    # If the script is run directly, execute your app or other logic
    test_input()
    test_output_shape()