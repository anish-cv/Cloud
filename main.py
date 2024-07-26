import numpy as np
from flask import Flask, request, jsonify, render_template, render_template_string, url_for
import pickle
import joblib


app = Flask(__name__)


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction1 = model_ricebug.predict(final_features)
    prediction2 =model_leaffolder.predict(final_features)
    prediction3 =model_gall_midge.predict(final_features)
    prediction4 =model_green_leaf.predict(final_features)

    #prediction5 = model_kanaka.predict(final_features)
    prediction6 = model_dhana.predict(final_features)
    #prediction7 = model_anakkayam.predict(final_features)
    prediction8 = model_madakkathara.predict(final_features)

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text=" Attack chance Ricebug {} : leaffolder {} : gall_midge {} : green_leaf{} : kanaka{} : dhana{}: anakkayam{}: madakkathara{} ".format(prediction1[0], prediction2[0], prediction3[0], prediction4[0], prediction5[0], prediction6[0], prediction7[0], prediction8[0]))

@app.route('/predict_api2',methods=['POST'])
def predict_api2():
    try:
        model_ricebug = joblib.load('RandomForest_Ricebug.pkl')
        model_leaffolder = joblib.load('XGBClassifier_LeafFolder.pkl')
        model_gall_midge = joblib.load('XGBClassifier_GallMidge.pkl')
        model_green_leaf = joblib.load('XGBClassifier_Greenleafhopper.pkl')

        data = request.json
        features = data['features']
    
        # Make prediction
        prediction1 = model_ricebug.predict([features])[0]
        prediction2 = model_leaffolder.predict([features])[0]
        prediction3 = model_gall_midge.predict([features])[0]
        prediction4 = model_green_leaf.predict([features])[0]     

        # Prepare response as JSON
        response = {'ricebug': int(prediction1), 'leaffolder': int(prediction2), 'gall_midge':int(prediction3), 'green_leaf':int(prediction4)}

        del model_ricebug
        del model_leaffolder
        del model_gall_midge
        del model_green_leaf


        return jsonify(response)
    except Exception as e:
        # Handle prediction error
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 500    
    #return prediction1

@app.route('/predict_api3',methods=['POST'])
def predict_api3():
    try:
        #model_kanaka = joblib.load('DecisionTree_Kanaka.pkl')
        model_dhana = joblib.load('gradiant_DHANA.pkl')
        #model_anakkayam = joblib.load('LogisticRegression_Anakkayam.pkl')
        model_madakkathara = joblib.load('XGB_MADAKKATHARA.pkl')

        data = request.json
        features = data['features']
    
        # Make prediction        
        #prediction5 = model_kanaka.predict([features])[0]        
        prediction6 = model_dhana.predict([features])[0]
        #prediction7 = model_anakkayam.predict([features])[0]
        prediction8 = model_madakkathara.predict([features])[0]

        # Prepare response as JSON
        response = { 'dhana':int(prediction6), 'madakkathara':int(prediction8)}

        del model_dhana
        del model_madakkathara
        
        return jsonify(response)
    except Exception as e:
        # Handle prediction error
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 500    
    #return prediction1

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)