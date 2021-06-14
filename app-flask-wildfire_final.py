from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import gzip, pickle, pickletools
import json

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################

from flask import jsonify


@app.route('/index')
def index():
    return flask.render_template('webpage_new.html')


@app.route('/background_process')
def background_process():
    
    LATITUDE = request.args.get('LATITUDE', 0, type=float)
    LONGITUDE = request.args.get('LONGITUDE', 0, type=float)
    DISCOVERY_DOY = request.args.get('DISCOVERY_DOY', 0, type=int)
    FIRE_SIZE = request.args.get('FIRE_SIZE', 0, type=int)
    STATE = request.args.get('STATE', 0, type=str)
    FIRE_YEAR = request.args.get('FIRE_YEAR', 0, type=int)
    MONTH = request.args.get('MONTH', 0, type=int)
    DAY_OF_WEEK = request.args.get('DAY_OF_WEEK', 0, type=str)
    FirePerMonth = request.args.get('FirePerMonth', 0, type=int)
    
    filepath = 'random_forest_2.pkl'
    with gzip.open(filepath, 'rb') as f:
        p = pickle.Unpickler(f)
        clf = p.load()
    #clf = joblib.load('model_1.pkl')
    LabelEncodeState = joblib.load('LabelEncodeState.pkl')
    LabelEncodeDayofweek = joblib.load('LabelEncodeDayofweek.pkl')
    NormalizeContinuousVariable = joblib.load('NormalizeContinuousVariable.pkl')
    
    #to_predict_list = request.form.to_dict()

    #review_text = clean_text(to_predict_list['review_text'])
    state = LabelEncodeState.transform(np.array(STATE).reshape(-1,1))
    Dayofweek = LabelEncodeDayofweek.transform(np.array(DAY_OF_WEEK).reshape(-1,1))


    normalize=NormalizeContinuousVariable.transform([[np.array(LATITUDE),
                                                    np.array(LONGITUDE),
                                                    np.array(DISCOVERY_DOY),
                                                    np.array(FIRE_SIZE),
                                                    np.array(FIRE_YEAR),
                                                    np.array(MONTH),
                                                    np.array(FirePerMonth)]])
    def Sort(listoflist):
        listoflist.sort(key = lambda x: x[1],reverse = True)
        return listoflist 

    Test_data = np.hstack((state,Dayofweek,normalize.flatten()))
    #pred=clf.predict(Test_data.reshape(1, -1))
    list_1=['Lightning','Equipment Use','Smoking','Campfire',
            'Debris Burning','Railroad','Arson','Children',
            'Miscellaneous','Fireworks','Powerline','Structure',
            'Missing/Undefined']
        
    proba= np.round(np.multiply(clf.predict_proba(Test_data.reshape(1, -1)),100),4)
    
    pred=Sort(list(zip(list_1,proba[0])))[0:5]
    
    
    return jsonify(result=json.dumps(pred))
  
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8080)
