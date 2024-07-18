import pandas as pd
import joblib

def preprocess_data(df, model_pipeline):

    preprocessor = model_pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(df)
    X_transformed_df = pd.DataFrame(X_transformed, columns=model_pipeline.named_steps['preprocessor'].get_feature_names_out())
    
    return X_transformed_df

new_customer = {

    'CustomerID': 'C001',
    'Age': 30,
    'Gender': 'Female',
    'Tenure': 12,
    'Usage Frequency': 25,
    'Support Calls': 3,
    'Payment Delay': 5,
    'Subscription Type': 'Premium',
    'Contract Length': 12,
    'Total Spend': 500,
    'Last Interaction': 10

}

new_customer_df = pd.DataFrame([new_customer])

model_pipeline = joblib.load('Logistic Regression.pkl')

preprocessed_new_customer = preprocess_data(new_customer_df, model_pipeline)

prediction = model_pipeline.named_steps['classifier'].predict(preprocessed_new_customer)
prediction_prob = model_pipeline.named_steps['classifier'].predict_proba(preprocessed_new_customer)

if prediction[0] == 1:
    print(f"Customer {new_customer['CustomerID']} is likely to churn.")
else:
    print(f"Customer {new_customer['CustomerID']} is not likely to churn.")

print(f"Churn Probability: {prediction_prob[0][1]:.2f}")
