import joblib

features = joblib.load("feature_names.pkl")

print("Jumlah fitur:", len(features))
print("Daftar fitur:")
print(features)