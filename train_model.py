import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_and_preprocess_data(csv_path="laptop_data.csv"):
    print("[1/5] Loading raw dataset...")
    df = pd.read_csv(csv_path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    print("[2/5] Cleaning, market calibrating, and feature engineering...")
    
    # Marketplace calibration factor:
    # Adjusts historical base prices to current Indian e-commerce marketplace levels (Flipkart & Amazon.in)
    df["Price"] = df["Price"] * 1.18

    # 1. Ram and Weight
    df["Ram"] = df["Ram"].str.replace("GB", "").astype("int32")
    df["Weight"] = df["Weight"].str.replace("kg", "").astype("float32")

    # 2. Touchscreen & IPS Display
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in str(x) else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in str(x) else 0)

    # 3. Calculate Pixels Per Inch (PPI)
    split_res = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["X_res"] = split_res[0].str.replace(",", "").str.findall(r"(\d+\.?\d+)").apply(lambda x: x[-1]).astype("int32")
    df["Y_res"] = split_res[1].astype("int32")
    df["Ppi"] = (((df["X_res"]**2) + (df["Y_res"]**2))**0.5 / df["Inches"]).astype("float32")
    df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"], inplace=True)

    # 4. CPU Brand Categorization
    def fetch_processor(text):
        words = text.split()[0:3]
        text_joined = " ".join(words)
        if text_joined in ["Intel Core i7", "Intel Core i5", "Intel Core i3"]:
            return text_joined
        elif text.split()[0] == "Intel":
            return "Other Intel Processor"
        else:
            return "AMD Processor"

    df["Cpu_brand"] = df["Cpu"].apply(fetch_processor)
    df.drop(columns=["Cpu"], inplace=True)

    # 5. Storage Decomposition (SSD & HDD in GB)
    df["Memory"] = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
    df["Memory"] = df["Memory"].str.replace("GB", "")
    df["Memory"] = df["Memory"].str.replace("TB", "000")
    new = df["Memory"].str.split("+", n=1, expand=True)

    df["first"] = new[0].str.strip()
    df["second"] = new[1].fillna("0").str.strip()

    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df["first"] = df["first"].str.extract(r"(\d+)").astype(int)

    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["second"] = df["second"].str.extract(r"(\d+)").fillna(0).astype(int)

    df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
    df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
    df.drop(columns=["first", "second", "Layer1HDD", "Layer1SSD", "Layer2HDD", "Layer2SSD", "Memory"], inplace=True)

    # 6. GPU Brand
    df["Gpu_brand"] = df["Gpu"].apply(lambda x: x.split()[0])
    df = df[df["Gpu_brand"] != "ARM"]  # Drop outlier ARM row
    df.drop(columns=["Gpu"], inplace=True)

    # 7. Operating System grouping
    def cat_os(inp):
        if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
            return "Windows"
        elif inp in ["macOS", "Mac OS X"]:
            return "Mac"
        else:
            return "Others/No OS/Linux"

    df["OpSys"] = df["OpSys"].apply(cat_os)

    print(f"Processed dataset shape: {df.shape}")
    return df

def train_and_evaluate(df):
    print("[3/5] Splitting features and target...")
    X = df.drop(columns=["Price"])
    y = np.log(df["Price"])  # Log transform target for normal distribution

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

    # Categorical column transformer: indices 0, 1, 3, 8, 11 (Company, TypeName, OpSys, Cpu_brand, Gpu_brand)
    step1 = ColumnTransformer(
        transformers=[
            ("col_tnf", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), [0, 1, 3, 8, 11])
        ],
        remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "Lasso Regression": Lasso(alpha=0.001),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=2),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, random_state=2),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=3,
            max_samples=0.5,
            max_features=0.75,
            max_depth=15
        )
    }

    print("\n[4/5] Model Benchmark Evaluation (Indian Marketplace Parity):")
    print("-" * 65)
    print(f"{'Model Name':<22} | {'R2 Score':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 65)

    best_score = -1.0
    best_pipe = None
    best_name = ""

    for name, model in models.items():
        pipe = Pipeline([
            ("step1", step1),
            ("step2", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"{name:<22} | {r2:.4f}     | {mae:.4f}     | {rmse:.4f}")

        if r2 > best_score:
            best_score = r2
            best_pipe = pipe
            best_name = name

    print("-" * 65)
    print(f"--> Best Performing Model: {best_name} with R2 = {best_score:.4f}")

    print("\n[5/5] Exporting model artifacts (df.pkl, pipe.pkl)...")
    with open("df.pkl", "wb") as f:
        pickle.dump(df, f)

    with open("pipe.pkl", "wb") as f:
        pickle.dump(best_pipe, f)

    print("Model training and export complete!")

if __name__ == "__main__":
    df = load_and_preprocess_data("laptop_data.csv")
    train_and_evaluate(df)
