# Save model and encoders
print("Saving model and encoders...")
joblib.dump(model, 'salary_prediction_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

print("✅ Model and encoders saved successfully!")

# Let's also test loading them to make sure they work
print("\nTesting model loading...")
loaded_model = joblib.load('salary_prediction_model.joblib')
loaded_encoders = joblib.load('label_encoders.joblib')

print("✅ Model and encoders loaded successfully!")
print(f"Model type: {type(loaded_model)}")
print(f"Number of encoders: {len(loaded_encoders)}")
print(f"Encoder keys: {list(loaded_encoders.keys())}")

# Test a prediction to make sure everything works
print("\nTesting a sample prediction...")

# Create a sample input
sample_input = pd.DataFrame([{
    'current_company': 'infosys',
    'target_company': 'google',
    'years_of_experience': 5.0,
    'current_salary': 1500000,
    'gender': 'Male',
    'location': 'Bangalore',
    'current_role': 'Software Developer',
    'sector': 'IT',
    'education': 'Bachelor',
    'certifications': 'AWS',
    'skills': 'Cloud',
    'current_company_tier': 3,
    'target_company_tier': 1
}])

print("Sample input:")
print(sample_input)

# Encode the sample input
sample_encoded = sample_input.copy()
for col in categorical_columns:
    if col in sample_input.columns:
        try:
            sample_encoded[col] = loaded_encoders[col].transform(sample_input[col].astype(str))
        except ValueError as e:
            print(f"Warning: {col} value not seen during training, using first class")
            sample_encoded[col] = 0

# Make prediction
prediction = loaded_model.predict(sample_encoded[feature_columns])
print(f"\nSample prediction: ₹{prediction[0]:,.0f}")

print("\n✅ Everything is working correctly!")
print("You can now run your Streamlit app with: streamlit run 3_streamlit_app.py")