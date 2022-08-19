import pickle
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/review', methods=['POST','GET']) # route to show the review comments in a web UI
@cross_origin()
def index():
    try:
        reviews = []
        saved_model = pickle.load(open('linr_model.sav', 'rb'))
        prediction = saved_model.predict([[ request.form['CRIM'],  request.form['ZN'], request.form['INDUS'],  request.form['CHAS'],
               request.form['NOX'],  request.form['RM'], request.form['AGE'], request.form['DIS'],
                request.form['RAD'], request.form['TAX'], request.form['PTRATIO'], request.form['B'] ,
               request.form['LSTAT']]])[0]
        return f'Prediction: {prediction}'

    except Exception as e:
        return f'Error message is : {e}'

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8002, debug=True)