from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Cargar el bundle (el que ya tienes generado)
bundle = joblib.load('spotify_api_bundle.pkl')
model = bundle['model']
artist_map = bundle['artist_map']
genre_map = bundle['genre_map']
album_map = bundle['album_map']
global_mean = bundle['global_mean']

try:
    feature_names = model.feature_names_in_.tolist()
except AttributeError:
    # Si por alguna razón no los encuentra, los extraemos del booster
    feature_names = model.get_booster().feature_names

@app.route('/', methods=['GET'])
def home():
    return "API Spotify Popularity Predictor - Online (Corregida)"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        
        # Transformación Target Encoding
        df_input['artists_enc'] = df_input['artists'].map(artist_map).fillna(global_mean)
        df_input['track_genre_enc'] = df_input['track_genre'].map(genre_map).fillna(global_mean)
        df_input['album_name_enc'] = df_input['album_name'].map(album_map).fillna(global_mean)
        
        # Convertir explicit a entero si viene como booleano o string
        if 'explicit' in df_input.columns:
            df_input['explicit'] = df_input['explicit'].astype(int)
            
        # 3. Reordenar las columnas según lo que espera el modelo
        X_api = df_input[feature_names]
        
        # 4. Predicción
        prediction = model.predict(X_api)[0]
        
        return jsonify({
            'status': 'success',
            'popularity': float(np.clip(prediction, 0, 100))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)