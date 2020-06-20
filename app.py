from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('demo-model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    int_features = [np.array(int_features)]
    salary = model.predict(int_features)

    output = round(salary[0],2)

    return render_template('index.html',prediction_text = 'Employee Salary should be {}'.format(output))


if __name__ == '__main__':
    app.run()
