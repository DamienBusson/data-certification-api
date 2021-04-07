
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict/")
def popularity_pred(acousticness,
             danceability,   # noqa: E128
             duration_ms,  # noqa: E128
             energy,  # noqa: E128
             explicit,   # noqa: E128
             id,   # noqa: E128
             instrumentalness,   # noqa: E128
             key,   # noqa: E128
             liveness,   # noqa: E128
             loudness,   # noqa: E128
             mode,   # noqa: E128
             name,     # noqa: E128
             release_date,   # noqa: E128
             speechiness,   # noqa: E128
             tempo,   # noqa: E128
             valence,   # noqa: E128
             artist):   # noqa: E128


    track_formatted = pd.DataFrame(dict(
        acousticness=[acousticness],
        danceability=[float(danceability)],
        duration_ms=[int(duration_ms)],
        energy=[float(energy)],
        explicit=[int(explicit)],
        id=[str(id)],
        instrumentalness=[float(instrumentalness)],
        key=[int(key)],
        liveness=[float(liveness)],
        loudness=[float(loudness)],
        mode=[int(mode)],
        name=[str(name)],
        release_date=[str(release_date)],
        speechiness=[float(speechiness)],
        tempo=[float(tempo)],
        valence=[float(valence)],
        artist=[str(artist)]
        ))

    model = joblib.load('model.joblib')

    popularity_prediction = model.predict(track_formatted)[0]

    return dict(artist=artist,
        song_name=name,
        prediction=popularity_prediction)


