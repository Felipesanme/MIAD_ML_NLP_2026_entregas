from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Configuración de la API y la interfaz Swagger
api = Api(app, 
          version='1.0', 
          title='Spotify Popularity Prediction API',
          description='Interfaz interactiva para predecir la popularidad de canciones')

# Cargar el bundle (asegúrate de que el nombre del archivo coincida)
bundle = joblib.load('spotify_api_bundle.pkl')
model = bundle['model']
artist_map = bundle['artist_map']
genre_map = bundle['genre_map']
album_map = bundle['album_map']
global_mean = bundle['global_mean']

# Intentar extraer los nombres de las variables del modelo
try:
    feature_names = model.feature_names_in_.tolist()
except AttributeError:
    feature_names = model.get_booster().feature_names

# Definición de los argumentos que aparecerán en la interfaz (Cajitas de texto)
parser = reqparse.RequestParser()
parser.add_argument('artists', type=str, required=True, help='Nombre del Artista', location='args')
parser.add_argument('track_genre', type=str, required=True, help='Género de la canción', location='args')
parser.add_argument('album_name', type=str, required=True, help='Nombre del Álbum', location='args')
parser.add_argument('duration_ms', type=int, required=True, help='Duración en ms', location='args')
parser.add_argument('explicit', type=int, required=True, help='Explícita (0 o 1)', location='args')
parser.add_argument('danceability', type=float, required=True, help='Danceability (0-1)', location='args')
parser.add_argument('energy', type=float, required=True, help='Energy (0-1)', location='args')
parser.add_argument('key', type=int, required=True, help='Key (0-11)', location='args')
parser.add_argument('loudness', type=float, required=True, help='Loudness (dB)', location='args')
parser.add_argument('mode', type=int, required=True, help='Mode (0 o 1)', location='args')
parser.add_argument('speechiness', type=float, required=True, help='Speechiness (0-1)', location='args')
parser.add_argument('acousticness', type=float, required=True, help='Acousticness (0-1)', location='args')
parser.add_argument('instrumentalness', type=float, required=True, help='Instrumentalness (0-1)', location='args')
parser.add_argument('liveness', type=float, required=True, help='Liveness (0-1)', location='args')
parser.add_argument('valence', type=float, required=True, help='Valence (0-1)', location='args')
parser.add_argument('tempo', type=float, required=True, help='Tempo (BPM)', location='args')
parser.add_argument('time_signature', type=int, required=True, help='Time Signature', location='args')

# Definición de la respuesta
resource_fields = api.model('Resource', {
    'popularity_predicted': fields.Float,
    'status': fields.String
})

@api.route('/predict')
class SpotifyApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        # Capturar los argumentos de las cajitas de texto
        args = parser.parse_args()
        df_input = pd.DataFrame([args])
        
        # Aplicar Target Encoding (usando los mapas cargados)
        df_input['artists_enc'] = df_input['artists'].map(artist_map).fillna(global_mean)
        df_input['track_genre_enc'] = df_input['track_genre'].map(genre_map).fillna(global_mean)
        df_input['album_name_enc'] = df_input['album_name'].map(album_map).fillna(global_mean)
        
        # Asegurar el orden de las columnas para el XGBoost
        X_api = df_input[feature_names]
        
        # Realizar predicción
        prediction = model.predict(X_api)[0]
        
        return {
            "popularity_predicted": float(np.clip(prediction, 0, 100)),
            "status": "success"
        }, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)