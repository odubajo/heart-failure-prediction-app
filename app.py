from flask import Flask,jsonify, render_template, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
pickle_ln= open('model.pkl','rb')
model=pickle.load(pickle_ln)


@app.route("/")
def home():
    return render_template("model.html")

@app.route("/predict",methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age= request.form['age']
    sex= request.form['sex']
    chest_pain_type= request.form['chest_pain_type']
    resting_bp= request.form['resting_bp']
    cholesterol= request.form['cholesterol']
    fasting_bs= request.form['fasting_bs']
    resting_ECG= request.form['resting_ECG']
    maxHR= request.form['maxHR']
    exercise_angina= request.form['exercise_angina']
    old_peak= request.form['old_peak']
    ST_slope= request.form['ST_slope']
    arr = np.array([[age,sex,chest_pain_type,resting_bp,cholesterol,fasting_bs,resting_ECG,maxHR,exercise_angina,old_peak,ST_slope]])
    prediction=model.predict(arr)
    if prediction == 0:
        return render_template("negative.html")
    else:
        return render_template("positive.html")
        
    
   

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
 
 