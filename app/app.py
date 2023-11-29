from flask import Flask, render_template, request,jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
loaded_model = joblib.load('./best_model.pkl')
loaded_scaler = joblib.load('./scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        resting_bp = float(request.form['resting_bp'])
        cholesterol = float(request.form['cholesterol'])
        max_heart_rate = float(request.form['max_heart_rate'])
        st_Depression = float(request.form['st_Depression'])

        # Convert binary dropdown values to 1 or 0
        sex_Male = int(request.form['sex_Male'])
        fasting_Blood_Sugar_True = int(request.form['fasting_Blood_Sugar_True'])
        chest_Pain_Type_atypical_angina = int(request.form['chest_Pain_Type_atypical_angina'])
        chest_Pain_Type_atypical_anginal = int(request.form['chest_Pain_Type_atypical_anginal'])
        resting_ECG_normal = int(request.form['resting_ECG_normal'])
        resting_ECG_st_t_wave_abnormality = int(request.form['resting_ECG_st_t_wave_abnormality'])
        exercise_Induced_Angina_True = int(request.form['exercise_Induced_Angina_True'])
        slope_of_ST_flat = int(request.form['slope_of_ST_flat'])
        thalassemia_Type_normal = int(request.form['thalassemia_Type_normal'])
        thalassemia_Type_reversible_defect = int(request.form['thalassemia_Type_reversible_defect'])
        
        # Add other binary features here...

        # Create a NumPy array with the input features
        # user_input = np.array([age, resting_bp, cholesterol, max_heart_rate, st_Depression, sex_Male,
        #                        f
        # asting_Blood_Sugar_True])
        # age = 30
        # resting_bp = 62
        # cholesterol = 343
        # max_heart_rate = 77
        # st_Depression = 0.1
        # sex_Male = 0
        # chest_Pain_Type_atypical_angina = 0
        # chest_Pain_Type_atypical_anginal = 0
        # fasting_Blood_Sugar_True = 1
        # resting_ECG_normal = 0
        # resting_ECG_st_t_wave_abnormality = 1
        # exercise_Induced_Angina_True = 0
        # slope_of_ST_flat = 1
        # thalassemia_Type_normal = 1
        # thalassemia_Type_reversible_defect = 0
        user_input = np.array([age, resting_bp, cholesterol, max_heart_rate, st_Depression, sex_Male, chest_Pain_Type_atypical_angina,
                       chest_Pain_Type_atypical_anginal, fasting_Blood_Sugar_True, resting_ECG_normal,
                       resting_ECG_st_t_wave_abnormality, exercise_Induced_Angina_True, slope_of_ST_flat,
                       thalassemia_Type_normal, thalassemia_Type_reversible_defect])
        print(user_input)
        # Separate the features to scale
        features_to_scale = user_input[:5]

        # Fit and transform the scaler on the training data
        scaled_features = loaded_scaler.transform(features_to_scale.reshape(1, -1))

        # Update the user_input with the scaled features
        user_input[:5] = scaled_features.flatten()
        print(user_input)
        # Now you can use the user_input for prediction
        prediction = loaded_model.predict(user_input.reshape(1, -1))[0]
        if prediction:
            prediction = "May Have Heart Disease."
        else:
            prediction = "No Heart Disease."

        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
