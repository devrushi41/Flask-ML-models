from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pickle

app = Flask(__name__)

infile1 = open('./spamsms', 'rb')
smsspammodel = pickle.load(infile1)

infile2 = open('./vectorizer', 'rb')
smsspamvectorizer = pickle.load(infile2)

infile3 = open('./knnmodel', 'rb')
knnmodel = pickle.load(infile3)

infile4 = open('./predictor', 'rb')
knnpredictor = pickle.load(infile4)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/spamsms', methods=["POST", "GET"])
def spamsms():
    try:
        if request.method == "POST":
            sms = request.form['message']
            predict_sms = smsspammodel.predict(
                smsspamvectorizer.transform([sms]))
            if(predict_sms[0] == 1):
                result = "The SMS is Spam"
            else:
                result = "The SMS is Ham"
            return result
        else:
            return render_template('spamsms.html')
    except:
        return 'There is a problem at our end'


@app.route('/fruitspredictor', methods=["POST", "GET"])
def fruitspredictor():
    try:
        if request.method == "POST":
            mass = request.form['mass']
            width = request.form['width']
            height = request.form["height"]
            arr = np.array([mass, width, height]).astype(np.float64)
            predict_fruit = knnmodel.predict([arr])
            result = 'The fruit is ' + knnpredictor[predict_fruit[0]]
            session['result'] = result
            return result
        else:
            return render_template('fruitpred.html')
    except:
        return 'there is a problem at our end'


if __name__ == "__main__":
    app.run()
