from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        prediction = model.predict(final_input)
        return render_template('index.html', prediction_text=f"🏠 Predicted Median House Value: ${prediction[0]*100000:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # use the port Render provides
    app.run(debug=True, host='0.0.0.0', port=port)  # Bind to 0.0.0.0 for public access
