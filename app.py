from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open('Linear_Model.pkl', 'rb') as f:
    model = pickle.load(f)


project_data = {
    "carat": None,
    "depth": None,
    "table": None,
    "x": None,
    "y": None,
    "z": None,
    "cut": {'Ideal': 5, 'Premium': 4, 'Very Good': 3, 'Good': 2, 'Fair': 1},
    "color": {'D': 7, 'E': 6, 'F': 5, 'G': 4, 'H': 3, 'I': 2, 'J': 1},
    "clarity": {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8},
    "columns": ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
}

def construct_test_array(project_data, carat, depth, table, x, y, z, cut, color, clarity):
    """Construct the test array for prediction."""
    test_array = np.zeros(len(project_data['columns']))

    # Populate the test array with the input values
    test_array[0] = carat
    test_array[1] = depth
    test_array[2] = table
    test_array[3] = x
    test_array[4] = y
    test_array[5] = z
    test_array[6] = project_data['cut'][cut]
    test_array[7] = project_data['color'][color]
    test_array[8] = project_data['clarity'][clarity]

    return test_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        carat = float(request.form['carat'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']

        # Construct the test array
        test_array = construct_test_array(project_data, carat, depth, table, x, y, z, cut, color, clarity)

        # Predict using the loaded model
        result = model.predict([test_array])[0]

        return jsonify({'predicted_price': round(result, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5005, debug=True)
