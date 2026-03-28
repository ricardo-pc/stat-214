import pandas as pd

# split files
train_features = pd.read_csv("../data/train_features.csv")
val_features = pd.read_csv("../data/val_features.csv")
test_features = pd.read_csv("../data/test_features.csv")

# embedding files by real image ID
train_ae = pd.read_csv("../data/ae_embeddings/O013257_ae.csv")
val_ae = pd.read_csv("../data/ae_embeddings/O012791_ae.csv")
test_ae = pd.read_csv("../data/ae_embeddings/O013490_ae.csv")

# sanity checks
print("Duplicate (x, y) in train_ae:", train_ae.duplicated(["x", "y"]).sum()) #0
print("Duplicate (x, y) in val_ae:", val_ae.duplicated(["x", "y"]).sum()) #0
print("Duplicate (x, y) in test_ae:", test_ae.duplicated(["x", "y"]).sum()) #0

# merge on x and y
train_model = train_features.merge(train_ae, on=["x", "y"], how="left")
val_model = val_features.merge(val_ae, on=["x", "y"], how="left")
test_model = test_features.merge(test_ae, on=["x", "y"], how="left")

# save
train_model.to_csv("../data/train_model.csv", index=False)
val_model.to_csv("../data/val_model.csv", index=False)
test_model.to_csv("../data/test_model.csv", index=False)

print("Saved:")
print("../data/train_model.csv", train_model.shape)
print("../data/val_model.csv", val_model.shape)
print("../data/test_model.csv", test_model.shape)

print("\nMissing values after merge:")
print("train:\n", train_model.isna().sum())
print("val:\n", val_model.isna().sum())
print("test:\n", test_model.isna().sum())