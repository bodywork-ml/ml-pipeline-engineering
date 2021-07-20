"""
- Get model and load into memory.
- Start web API server.
"""
import sys
from enum import Enum
from typing import Dict, Union

import uvicorn
from bodywork_pipeline_utils import aws, logging
from fastapi import FastAPI, status
from numpy import array
from pydantic import BaseModel, Field

from pipeline.train_model import PRODUCT_CODE_MAP

app = FastAPI(debug=False)
log = logging.configure_logger()


class ProductCode(Enum):
    SKU001 = "SKU001"
    SKU002 = "SKU002"
    SKU003 = "SKU003"
    SKU004 = "SKU004"
    SKU005 = "SKU005"


class Data(BaseModel):
    product_code: ProductCode
    orders_placed: float = Field(..., ge=0.0)


class Prediction(BaseModel):
    est_hours_to_dispatch: float
    model_version: str


@app.post(
    "/api/v0.1/time_to_dispatch",
    status_code=status.HTTP_200_OK,
    response_model=Prediction,
)
def time_to_dispatch(data: Data) -> Dict[str, Union[str, float]]:
    features = array([[data.orders_placed, PRODUCT_CODE_MAP[data.product_code.value]]])
    prediction = wrapped_model.model.predict(features).tolist()[0]
    return {"est_hours_to_dispatch": prediction, "model_version": str(wrapped_model)}


if __name__ == "__main__":
    try:
        args = sys.argv
        s3_bucket = args[1]
        wrapped_model = aws.get_latest_pkl_model_from_s3(s3_bucket, "models")
        log.info(f"Successfully loaded model: {wrapped_model}")
    except IndexError:
        log.error("Invalid arguments passed to serve_model.py - expected S3_BUCKET")
        sys.exit(1)
    except Exception as e:
        log.error(f"Could not get latest model and start web server - {e}")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", workers=1)
