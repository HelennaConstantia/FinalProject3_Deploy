from logging import debug
from flask import Flask, render_template, request
import numpy as np
import pickle
model=pickle.load(open('model_RF.pkl','rb'))
app=Flask(__name__,template_folder='templates')
@app.route('/')
def form():
    return render_template("main.html")
@app.route('/predict',methods=["POST"])
def predict():
    age=float(request.form['age'])
    creatin=float(request.form['creatin'])
    ejection=float(request.form['ejection'])
    platelets=float(request.form['platelets'])
    serum_c=float(request.form['serum_c'])
    serum_s=float(request.form['serum_s'])
    time=float(request.form['time'])
    int_features=[age,creatin,ejection,platelets,serum_c,serum_s,time]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output={0:'tidak meninggal', 1:'meninggal'}
    return render_template('main.html', prediction_text='Pasien penderita penyakit jantung {}'.format(output[prediction[0]]))
if __name__=='__main__':
    app.run(debug=True)
