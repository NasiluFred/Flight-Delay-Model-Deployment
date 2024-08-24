from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('ridge.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        form_values = list(request.form.values())
        
        # Filter out non-numeric values and handle them appropriately
        numeric_features = []
        for value in form_values:
            try:
                numeric_features.append(float(value))
            except ValueError:
                # Handle or log non-numeric values as needed
                pass
        
        if not numeric_features:
            return render_template('index.html', prediction_text="Invalid input: No numeric values found.")

        # Pad with zeros to match the required number of features
        #while len(numeric_features) < 31:
            #numeric_features.append(0.0)

        final_features = [np.array(numeric_features)]
        prediction = model.predict(final_features)
        
        return render_template('index.html', prediction_text="Flight Delay Prediction: {}".format(prediction[0]))
    
    except Exception as e:
        return render_template('index.html', prediction_text="Error: {}".format(str(e)))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        numeric_values = []
        
        for value in data.values():
            try:
                numeric_values.append(float(value))
            except ValueError:
                # Handle or log non-numeric values as needed
                pass
        
        if not numeric_values:
            return jsonify({'error': 'Invalid input: No numeric values found.'})

        # Pad with zeros to match the required number of features
        #while len(numeric_values) < 31:
         #   numeric_values.append(0.0)

        prediction = model.predict([np.array(numeric_values)])
        
        return jsonify(prediction[0].tolist())
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
