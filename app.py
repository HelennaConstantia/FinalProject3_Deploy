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
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output={0:'survive', 1:'not survive'}
    return render_template('main.html', prediction_text='Pasien penderita penyakit jantung {}'.format(output[prediction[0]]))
if __name__=='__main__':
    app.run(debug=True)
