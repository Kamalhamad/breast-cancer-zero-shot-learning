import pandas as pd

from bczsl.data_processing import preprocess_image_paths, stratified_split


def test_preprocess_image_paths_returns_values():
    df = pd.DataFrame({"path": ["a.png", "b.png"]})
    out = preprocess_image_paths(df, "path")
    assert list(out) == ["a.png", "b.png"]


def test_stratified_split_preserves_classes():
    df = pd.DataFrame(
        {
            "label": ["benign"] * 10 + ["malignant"] * 10,
            "path": [f"img_{idx}.png" for idx in range(20)],
        }
    )
    train_df, test_df = stratified_split(df, "label", test_size=0.2)
    assert set(train_df["label"].unique()) == {"benign", "malignant"}
    assert set(test_df["label"].unique()) == {"benign", "malignant"}
