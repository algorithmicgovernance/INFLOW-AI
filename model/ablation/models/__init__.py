from .elastic_net_model import ElasticNetAblationModel
from .gradient_boosting_model import GradientBoostingAblationModel
from .linear_regression import LinearRegressionAblationModel
from .random_forest_model import RandomForestAblationModel
from .transformer_model import TransformerAblationModel

TEMPORAL_MODEL_REGISTRY = {
    RandomForestAblationModel.model_type: RandomForestAblationModel,
    GradientBoostingAblationModel.model_type: GradientBoostingAblationModel,
    LinearRegressionAblationModel.model_type: LinearRegressionAblationModel,
    ElasticNetAblationModel.model_type: ElasticNetAblationModel,
    TransformerAblationModel.model_type: TransformerAblationModel,
}

MODEL_REGISTRY = TEMPORAL_MODEL_REGISTRY

__all__ = ["TEMPORAL_MODEL_REGISTRY", "MODEL_REGISTRY"]
