import pandas as pd
from sklearn.model_selection import train_test_split


def load_metadata(csv_path):
    """Load dataset metadata."""
    return pd.read_csv(csv_path)


def stratified_split(df, label_column, test_size=0.2):
    """Perform stratified train-test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=42,
    )
    return train_df, test_df


def preprocess_image_paths(df, image_column):
    """Extract image paths from dataframe."""
    return df[image_column].values
