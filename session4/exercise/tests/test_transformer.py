from src.transform import Transformer, balance_dataset
import pandas as pd


def test_preprocess_binary_variables():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "HasCrCard": [1.0, 0.0, 1.0, 0.0],
            "IsActiveMember": [1.0, 0.0, 0.0, 1.0],
            "Exited": [1.0, 0.0, 1.0, 0.0],
            "Gender": ["Female", "Male", "Female", "Male"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "HasCrCard": [1, 0, 1, 0],
            "IsActiveMember": [1, 0, 0, 1],
            "Exited": [1, 0, 1, 0],
            "Gender": [1, 0, 1, 0],
        }
    )

    transformed_df = transformer._preprocess_binary_variables(df)

    pd.testing.assert_frame_equal(transformed_df, expected_df)


def test_transform_drops_ids_and_encodes_geography():
    transformer = Transformer()
    df = pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [100, 101],
            "Surname": ["Smith", "Jones"],
            "Geography": ["France", "Germany"],
            "Gender": ["Female", "Male"],
            "HasCrCard": [1.0, 0.0],
            "IsActiveMember": [0.0, 1.0],
            "Exited": [1.0, 0.0],
        }
    )

    transformed_df = transformer.transform(df)

    assert "RowNumber" not in transformed_df.columns
    assert "CustomerId" not in transformed_df.columns
    assert "Surname" not in transformed_df.columns
    assert "Geography" not in transformed_df.columns
    assert transformed_df["Gender"].tolist() == [1, 0]
    assert transformed_df["HasCrCard"].dtype == int
    assert transformed_df["Exited"].dtype == int
    assert transformed_df.filter(like="Geography_").shape[1] == 1


def create_df_balance():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50],
            "job": [
                "admin",
                "technician",
                "admin",
                "technician",
                "admin",
                "technician",
            ],
            "Exited": [0, 1, 0, 1, 0, 1],
        }
    )


def test_balance_dataset():
    df_balance = create_df_balance()
    expected_df = df_balance.sort_values(["age", "job"]).reset_index(drop=True)

    balanced_df = balance_dataset(df_balance)
    balanced_df = balanced_df.sort_values(["age", "job"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        balanced_df.sort_index(axis=1), expected_df.sort_index(axis=1)
    )


def test_balance_with_unequal_classes():
    df_unequal = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "job": ["admin", "technician", "admin", "technician", "admin"],
            "Exited": [0, 0, 1, 0, 1],
        }
    )

    balanced_df = balance_dataset(df_unequal)

    class_counts = balanced_df["Exited"].value_counts()
    assert class_counts[0] == class_counts[1]
