from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # Import pandas

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('carbon_emission_model.pkl', 'rb'))

# Define the column names for the model
columns = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
           'Transport', 'Vehicle Type', 'Social Activity', 'Monthly Grocery Bill',
           'Frequency of Traveling by Air', 'Vehicle Monthly Distance Km', 'Waste Bag Size',
           'Waste Bag Weekly Count', 'How Long TV PC Daily Hour', 'How Many New Clothes Monthly',
           'How Long Internet Daily Hour', 'Energy efficiency', 'Recycling', 'Cooking_With']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        body_type = request.form['body_type']
        sex = request.form['sex']
        diet = request.form['diet']
        shower = request.form['shower']
        energy_source = request.form['energy_source']
        transport = request.form['transport']
        vehicle_type = request.form['vehicle_type']
        social_activity = request.form['social_activity']
        grocery_bill = request.form['grocery_bill']
        air_travel = request.form['air_travel']
        vehicle_distance = request.form['vehicle_distance']
        waste_bag_size = request.form['waste_bag_size']
        waste_bag_count = request.form['waste_bag_count']
        tv_pc_hours = request.form['tv_pc_hours']
        new_clothes = request.form['new_clothes']
        internet_hours = request.form['internet_hours']
        energy_efficiency = request.form['energy_efficiency']
        recycling = request.form['recycling']
        cooking_with = request.form['cooking_with']

        # Create a DataFrame from the input data
        data = pd.DataFrame([[body_type, sex, diet, shower, energy_source, transport, vehicle_type,
                              social_activity, grocery_bill, air_travel, vehicle_distance, waste_bag_size,
                              waste_bag_count, tv_pc_hours, new_clothes, internet_hours, energy_efficiency,
                              recycling, cooking_with]], columns=columns)

        # Predict carbon emission using the model
        prediction = model.predict(data)[0]

        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)