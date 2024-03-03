from pydantic import BaseModel


class BirdSpeciePrediction(BaseModel):
    specie_name: str
    score: float


class Prediction(BaseModel):
    specie: BirdSpeciePrediction | None = None
    top_k: list[BirdSpeciePrediction] = list()
