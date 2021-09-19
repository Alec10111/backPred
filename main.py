from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# cargamos el modelo
pipe = load(r'C:\\Users\\Alec\\OneDrive\\Documentos\\Programming Stuff\\HousePrice_Pred\\src\\model_1.joblib')


def get_prediction(params):
    x = [[params.YearBuilt, params.TotalBath, params.BedroomAbvGr, params.YearRemodAdd]]
    y = pipe.predict(x)[0]  # just get single value
    return {'prediction': y}


# initiate API
app = FastAPI()

# origins = ["*"]
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Definimos una clase anotando los tipos de las features
class ModelFeatures(BaseModel):
    TotalBath: int
    BedroomAbvGr: int
    YearRemodAdd: int
    YearBuilt: int


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


@app.get("/new")
def read_root():
    return {"Hello": "ZM"}




@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/predict")
def predict(params: ModelFeatures):
    prediction = get_prediction(params)
    return prediction
