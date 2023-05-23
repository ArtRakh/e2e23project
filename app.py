from flask import Flask, request
import joblib
import numpy
import pandas

MODEL_PATH = 'mlmodels/model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route('/predict_price', methods = ['GET'])
def predict():
    args = request.args
    open_plan = args.get('open_plan', default=-1, type=int)
    first_day_exposition = args.get('first_day_exposition', default=-1, type=object)
    last_day_exposition = args.get('last_day_exposition', default=-1, type=object)
    rooms = args.get('rooms', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=int)

    df = pandas.DataFrame([first_day_exposition, last_day_exposition, open_plan, rooms, area, renovation])
    df['length_exposition'] = (pandas.to_datetime(df.last_day_exposition) - pandas.to_datetime(df.first_day_exposition)).dt.days
    df = pandas.get_dummies(df, columns=['open_plan', 'rooms', 'renovation'])
    df.drop(columns=['first_day_exposition', 'last_day_exposition'], inplace=True)
    x = numpy.array(df).reshape(1,-1)
    x = sc_x.transform(x)
    result = model.predict(x)
    result = sc_y.inverse_transform(result.reshape(1,-1))

    return str(result[0][0])


if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')
