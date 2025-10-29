import pandas as pd
import numpy as np
import random
import os
import uuid
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

def simulated_data(n_samples, fraud_ratio=0.03, harder=True, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = []
    for _ in range(n_samples):
        is_fraud = 1 if random.random() < fraud_ratio else 0

        type_ = random.choices(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
                               weights=[0.4, 0.4, 0.1, 0.05, 0.05])[0]

        if harder and is_fraud:
            type_ = random.choices(["TRANSFER", "CASH_OUT", "PAYMENT"], weights=[0.5, 0.3, 0.2])[0]
            amount = abs(np.random.normal(loc=800, scale=300))
            oldbalanceOrg = abs(np.random.normal(loc=2000, scale=500))
        else:
            amount = abs(np.random.exponential(scale=3000 if is_fraud else 1000))
            oldbalanceOrg = max(amount + abs(np.random.normal(0, 100)), 0)

        newbalanceOrig = max(oldbalanceOrg - amount, 0)
        oldbalanceDest = abs(np.random.exponential(2000))
        transfer_fraction = np.random.uniform(0.2, 0.9) if harder and is_fraud else 1.0
        newbalanceDest = oldbalanceDest + amount * transfer_fraction
        transaction_id = str(uuid.uuid4())

        data.append({
            "transactionID": transaction_id,
            "type": type_,
            "amount": round(amount, 2),
            "oldbalanceOrg": round(oldbalanceOrg, 2),
            "newbalanceOrig": round(newbalanceOrig, 2),
            "oldbalanceDest": round(oldbalanceDest, 2),
            "newbalanceDest": round(newbalanceDest, 2),
            "isFraud": is_fraud
        })

    return pd.DataFrame(data)

def generate_datasets(output_dir="datasets", num_sets=5, total_samples=20000, train_size=14000, fraud_ratio=0.02):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, num_sets + 1):
        df = simulated_data(n_samples=total_samples, fraud_ratio=fraud_ratio, harder=True, seed=42+i)

        train_df = df.sample(n=train_size, random_state=42+i)
        test_df = df.drop(train_df.index)

        test_df_hidden = test_df.drop(columns=["isFraud"])
        test_labels = test_df[["transactionID", "isFraud"]]

        train_df.to_csv(f"{output_dir}/train_dataset_{i}.csv", index=False)
        test_df_hidden.to_csv(f"{output_dir}/test_dataset_{i}.csv", index=False)
        test_labels.to_csv(f"{output_dir}/test_labels_{i}.csv", index=False)

    print(f"✅ Generated {num_sets} datasets with 14K train and 6K test in '{output_dir}'")

if __name__ == "__main__":
    generate_datasets()
