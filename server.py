# importing important libraries
from flask import Flask, request, render_template, flash, session
# importing file in which our ml-algorithms are residing
from models import Model
import os

app = Flask(__name__)


@app.route('/')
def root():
    # if not session.get('logged_in'):
    #     return render_template('login.html')
    # else:
    return render_template('index.html')


# @app.route('/login', methods=['POST'])
# def do_admin_login():
#     if request.form['password'] == 'test' and request.form['username'] == 'test':
#         session['logged_in'] = True
#     else:
#         flash('wrong password!')
#     return root()


# @app.route("/logout")
# def logout():
#     session['logged_in'] = False
#     return root()


@app.route('/predict', methods=["POST"])
def predict():
    # getting values from the form
    # answers of each questions
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    q10 = int(request.form['a10'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    # creating Model instance
    model = Model()
    # choosing algorithm
    classifier = model.svm_classifier()
    # predicting answer
    prediction = classifier.predict([values])
    # classification of our prediction
    if prediction[0] == 0:
        result = 'Test result : No Depression'
        return render_template("result1.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])
    if prediction[0] == 1:
        result = 'Test result : Mild Depression'
        return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])
    if prediction[0] == 2:
        result = 'Test result : Moderate Depression'
        return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])
    if prediction[0] == 3:
        result = 'Test result : Moderately severe Depression'
        return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])
    if prediction[0] == 4:
        result = 'Test result : Severe Depression'
        return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10])


@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')


@app.route('/contactus')
def Contact_us():
    return render_template('contact2.html')


app.secret_key = os.urandom(12)
app.run(port=5987, host='0.0.0.0', debug=True)
