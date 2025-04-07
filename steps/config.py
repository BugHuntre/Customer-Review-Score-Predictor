from pydantic import BaseModel, validator

class ModelNameConfig(BaseModel):
    model_name: str

    @validator("model_name")
    def validate_model_name(cls, value):
        if value != "LightGBM":
            raise ValueError("Only 'LightGBM' is supported in the current configuration.")
        return value
