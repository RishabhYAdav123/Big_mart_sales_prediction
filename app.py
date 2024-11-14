from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\kalpe\Desktop\Ml_project\best_rf.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numeric fields
        item_weight = float(request.form['Item_Weight'])
        item_visibility = float(request.form['Item_Visibility'])
        item_mrp = float(request.form['Item_MRP'])
        outlet_year = int(request.form['Outlet_Establishment_Year'])

        # Categorical fields - convert to numeric encoding (simple mapping)
        item_fat_content = request.form['Item_Fat_Content']
        item_type = request.form['Item_Type']
        outlet_size = request.form['Outlet_Size']
        outlet_location = request.form['Outlet_Location_Type']
        outlet_type = request.form['Outlet_Type']

        # Mapping categorical values to numerical values (adjust the mappings based on your model training)
        fat_content_mapping = {'Low Fat': 0, 'Regular': 1}
        item_fat_content = fat_content_mapping.get(item_fat_content, -1)  # Default to -1 if not found
        
        type_mapping = {'Dairy': 0, 'Fruits and Vegetables': 1, 'Snack Foods': 2, 'Frozen Foods': 3, 'Canned': 4}
        item_type = type_mapping.get(item_type, -1)  # Default to -1 if not found
        
        outlet_size_mapping = {'Small': 0, 'Medium': 1, 'High': 2}
        outlet_size = outlet_size_mapping.get(outlet_size, -1)  # Default to -1 if not found
        
        location_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        outlet_location = location_mapping.get(outlet_location, -1)  # Default to -1 if not found
        
        outlet_type_mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
        outlet_type = outlet_type_mapping.get(outlet_type, -1)  # Default to -1 if not found

        # Create a DataFrame with the input data to pass it to the model
        input_data = pd.DataFrame([{
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Establishment_Year': outlet_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location,
            'Outlet_Type': outlet_type
        }])

        # Make the prediction
        prediction = model.predict(input_data)

        # Return the predicted value
        return render_template('index.html', prediction_text=f'Predicted Item Outlet Sales: {prediction[0]}')

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
