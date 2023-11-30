from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the model and scaler from files
loaded_model = joblib.load('best_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Job title options
job_title_options = {
    'Accountant': 0, 'Back end Developer': 1, 'Business Analyst': 2, 'Business Development Associate': 3,
    'Business Development Manager': 4, 'Content Marketing Manager': 5, 'Customer Service Representative': 6,
    'Data Analyst': 7, 'Data Scientist': 8, 'Delivery Driver': 9, 'Digital Marketing Manager': 10,
    'Digital Marketing Specialist': 11, 'Director of Data Science': 12, 'Director of HR': 13,
    'Director of Marketing': 14, 'Director of Operations': 15, 'Financial Advisor': 16, 'Financial Analyst': 17,
    'Financial Manager': 18, 'Front End Developer': 19, 'Front end Developer': 20, 'Full Stack Engineer': 21,
    'Graphic Designer': 22, 'HR Coordinator': 23, 'HR Generalist': 24, 'HR Manager': 25,
    'Human Resources Coordinator': 26, 'Human Resources Manager': 27, 'Marketing Analyst': 28,
    'Marketing Coordinator': 29, 'Marketing Director': 30, 'Marketing Manager': 31, 'Marketing Specialist': 32,
    'Operations Analyst': 33, 'Operations Coordinator': 34, 'Operations Manager': 35, 'Other': 36,
    'Product Designer': 37, 'Product Manager': 38, 'Product Marketing Manager': 39, 'Project Coordinator': 40,
    'Project Engineer': 41, 'Project Manager': 42, 'Receptionist': 43, 'Research Director': 44,
    'Research Scientist': 45, 'Sales Associate': 46, 'Sales Director': 47, 'Sales Executive': 48,
    'Sales Manager': 49, 'Sales Representative': 50, 'Social Media Manager': 51, 'Software Developer': 52,
    'Software Engineer': 53, 'Software Engineer Manager': 54, 'UX Designer': 55, 'Web Developer': 56
}

# Type title options
type_title_options = {
    'hr': 0, 'design': 1, 'management': 2, 'engineer': 3, 'technical': 4, 'business': 5, 'operation': 6
}

# Education level options
education_level_options = {
    'High School': 0, 'Bachelor': 1, 'Master Degree': 2, 'PhD': 3
}

@app.route('/')
def index():
    return render_template('index.html', prediction_text="", default_experience=0, default_job_title_order="", default_education_level="", default_type_title="", job_title_options=job_title_options, type_title_options=type_title_options, education_level_options=education_level_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        experience = request.form['experience']
        job_title_order = request.form['job_title_order']
        education_level = request.form['education_level']
        type_title = request.form['type_title']

        # Create a NumPy array with the input features
        input_features = np.array([[float(experience), int(job_title_options[job_title_order]), int(education_level_options[education_level]), int(type_title_options[type_title])]])

        # Scale only the "Years of Experience" feature using the loaded scaler
        input_features[:, 0] = loaded_scaler.transform(input_features[:, 0].reshape(-1, 1)).flatten()
        
        # Make predictions using the loaded model
        predicted_salary = loaded_model.predict(input_features)

        # Print or use the prediction as needed
        result = np.exp(predicted_salary[0])


        prediction_text = f"Predicted Salary is {result:,.2f} $"

        # Pass the experience and dropdown values back to the form for display
        return render_template('index.html', prediction_text=prediction_text, default_experience=experience,
                            default_job_title_order=job_title_order, default_education_level=education_level, default_type_title=type_title, job_title_options=job_title_options, type_title_options=type_title_options, education_level_options=education_level_options)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        # Pass the experience and dropdown values back to the form for display
        return render_template('index.html', prediction_text=error_message, default_experience=experience,
                            default_job_title_order=job_title_order, default_education_level=education_level, default_type_title=type_title, job_title_options=job_title_options, type_title_options=type_title_options, education_level_options=education_level_options)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
