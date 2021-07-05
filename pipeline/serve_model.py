"""
- Get model and load into memory.
- Start web API server.
"""
from typing import Dict, Union

import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel

app = FastAPI(debug=False)


class Data(BaseModel):
    product_code: str
    orders_placed: float


class Prediction(BaseModel):
    est_hours_to_dispatch: float
    model_version: str


@app.post(
    "/api/v0.1/time_to_dispatch",
    status_code=status.HTTP_200_OK,
    response_model=Prediction,
)
def time_to_dispatch(data: Data) -> Dict[str, Union[str, float]]:
    return {"est_hours_to_dispatch": 1.0, "model_version": "0.0.1"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", workers=1)
