
from flask import Flask, render_template, request
import pickle
import numpy as np 

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    feature1 = float(request.form['Age in range 2 to 96'])
    feature2 = float(request.form['Sex 0 for male and 1 for female'])
    feature3 = float(request.form['T3 value in range 1 to 12'])
    feature4 = float(request.form['TT4 in range 3 to 431'])
    feature5 = float(request.form['T4U in range 1 to 3'])
    feature6 = float(request.form['FTI 3 to 396'])
    feature7 = float(request.form['referral_source_STMW 1 or 0'])
    feature8 = float(request.form['referral_source_SVHC 1 or 0'])
    feature9 = float(request.form['referral_source_SVHD 1 or 0'])
    feature10 = float(request.form['referral_source_SVI 1 or 0'])
    feature11 = float(request.form['referral_source_other 1 or 0'])
    
    pred = [feature1, feature2, feature3, feature4, 
                                 feature5, feature6, feature7,
                                 feature8, feature9, feature10,
                                 feature11]
    
    print(pred)

    prediction = model.predict(pred)
    final_prediction = np.where(prediction >= 0.49, 1, 0)
    
    if final_prediction == 0:
        result = "Good news! You are free from thyroidal disease."
    else:
        result = 'Our model has predicted that you have thyroidal disease.'
        
    return render_template('index.html', predict=result)

if __name__ == '__main__':
    app.run(debug=True)
    