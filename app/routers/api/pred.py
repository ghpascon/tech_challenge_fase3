from fastapi import APIRouter

from fastapi.responses import JSONResponse
from app.core.path import get_prefix_from_path
from app.schemas.ml import PredRequest, predictions

router_prefix = get_prefix_from_path(__file__)
router = APIRouter(prefix=router_prefix, tags=[router_prefix])

@router.post("/pred_loan_classification")
async def pred_loan_classification(pred_request: PredRequest):
    # Converte o Pydantic model para dict
    data = pred_request.model_dump()

    # Obtém a predição
    pred = predictions.predict(data)

    print(f"Predição: {pred}")

    # Retorna a predição como JSON
    return {"pred": pred[0]}
