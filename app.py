from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder='Templates')

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/Ai')
def Ai():
    return render_template('AI.html')

@app.route('/About')
def About():
    return render_template('About.html')

@app.route('/Services')
def Services():
    return render_template('Services.html')

@app.route('/Ai/spamAPP')
def spamClassifier():
    return render_template('SpamApp.html')

@app.route('/Ai/spamClassifier/result', methods=['POST'])
def spamClassifierResult():
    filename = 'nlp_model.pkl'
    clf = pickle.load(open(filename, 'rb'))
    cv = pickle.load(open('tranform.pkl', 'rb'))
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('SpamAppResult.html', prediction = my_prediction)

@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

if __name__ == "__main__":
    app.run()
