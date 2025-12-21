import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    df = df.drop_duplicates()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('species', axis=1))

    df_scaled = pd.DataFrame(
        X_scaled,
        columns=df.columns[:-1]
    )
    df_scaled['species'] = df['species'].values

    df_scaled.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("../iris_raw/iris_raw.csv", "iris_preprocessing.csv")
