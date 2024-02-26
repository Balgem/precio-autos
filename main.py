from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd


app = FastAPI()
model = pickle.load(open("model_reg.pkl", "rb"))

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return float(prediction[0])

@app.get("/")
async def root():
    return {"message": "Prediction"}

@app.get("/predict")
async def predict(
    Present_Price: float,
    Fuel_Type: int,
    Seller_Type: int,
    Transmission: int,
    Owner: int,
    logAge: float,
    logKMSDriven: float,
    PrecenPriceyLogAge: float,
    PrecentPriceyFuel: float,
    ):

    prediction = model.predict(
        [[Present_Price, Fuel_Type, Seller_Type, Transmission, Owner, logAge, logKMSDriven, PrecenPriceyLogAge, PrecentPriceyFuel]]
    )
    f"El precio del vehiculo es: {prediction}"

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")

#env\scripts\activate
#uvicorn main:app --reload
#pip install pipreqs
#pip freeze > requirements.txt
    
#