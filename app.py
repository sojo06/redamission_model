from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and Quantile Transformer
model = joblib.load("./readmission_model1.pkl")
quantile_transformer = joblib.load("./quantile_transformer.pkl")

def preprocess_input(data):
    """
    Process input dictionary into a DataFrame and compute required transformations.
    """
    # Convert input JSON into DataFrame
    input_df = pd.DataFrame([data])

    # Compute interaction feature
    input_df["Age_Stay_Interaction"] = input_df["age"] * input_df["length_of_stay"]

    # Apply Quantile Transformation
    input_df["Age_Stay_Interaction_Quantile"] = quantile_transformer.transform(input_df[["Age_Stay_Interaction"]])

    # Drop the original interaction feature (if it was dropped in training)
    input_df.drop(columns=["Age_Stay_Interaction"], inplace=True)

    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to get readmission prediction.
    """
    try:
        # Get JSON data from request
        data = request.json
        print(data)
        processed_data = preprocess_input(data)
        print("profefefefe")
        print(processed_data)
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Return result as JSON
        return jsonify({"readmission_prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
