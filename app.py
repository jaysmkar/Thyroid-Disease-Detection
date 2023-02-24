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
    features = [x for x in request.form.values()]
    
    
    form_data = {}
    form_data['feature1'] = request.form['Age in range 2 to 96']
    form_data['feature2'] = request.form['Sex 0 for male and 1 for female']
    form_data['feature3'] = request.form['T3 value in range 1 to 12']
    form_data['feature4'] = request.form['TT4 in range 3 to 431']
    form_data['feature5'] = request.form['T4U in range 1 to 3']
    form_data['feature6'] = request.form['FTI 3 to 396']
    form_data['feature7'] = request.form['referral_source_STMW 1 or 0']
    form_data['feature8'] = request.form['referral_source_SVHC 1 or 0']
    form_data['feature9'] = request.form['referral_source_SVHD 1 or 0']
    form_data['feature10'] = request.form['referral_source_SVI 1 or 0']
    form_data['feature11'] = request.form['referral_source_other 1 or 0']
    
    
    features_np = [np.array(form_data)]
    
    print(features_np)
    
    pred = model.predict(features_np)

    prediction = pred
    prediction = np.where(prediction>=0.49, 1, 0)
    
    if prediction == 0:
        result = "Good News! You are free from thyroidal disease."
    elif prediction == 1:
       result = "Our model has predicted that you have thyroidal disease."
        
    return render_template("index_thyroid.html", predict=result)  


if __name__ == "__main__":
    app.run(debug=True)  