MODELS_FOLDER = "session4/models"
DATASETS_FOLDER = "session4/exercise/dataset"
MODEL_NAME = "logistic-regression-model"

COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
BINARY_FEATURES = [
    "HasCrCard",
    "IsActiveMember",
    "Exited",
]
ONE_HOT_ENCODE_COLUMNS = [
    "Geography",
]
MODEL_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

