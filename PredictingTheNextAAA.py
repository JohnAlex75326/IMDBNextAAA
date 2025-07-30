import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import gzip

# Load title.basics and title.ratings
title_basics = pd.read_csv("title.basics.tsv.gz", sep="\t", na_values="\\N", dtype=str, compression="gzip")
title_ratings = pd.read_csv("title.ratings.tsv.gz", sep="\t", dtype={"tconst": str, "averageRating": float, "numVotes": int}, compression="gzip")

# Filter to movies with valid year and genres
title_basics = title_basics[title_basics["titleType"] == "movie"]
title_basics = title_basics.dropna(subset=["startYear", "genres"])
title_basics["startYear"] = pd.to_numeric(title_basics["startYear"], errors="coerce")
title_basics = title_basics[(title_basics["startYear"] >= 2010) & (title_basics["startYear"] <= 2024)]

# Merge
df = pd.merge(title_basics, title_ratings, on="tconst")
df = df.dropna(subset=["averageRating", "numVotes"])

# Compute Bayesian Average (WSS = Bayesian adjusted rating)
C = df["averageRating"].mean()
m = df["numVotes"].quantile(0.75)  # Tuneable: votes threshold
df["WSS"] = (df["numVotes"] / (df["numVotes"] + m)) * df["averageRating"] + (m / (df["numVotes"] + m)) * C

# Label as AAA title if in top 5% of WSS
threshold = df["WSS"].quantile(0.95)
df["isAAA"] = (df["WSS"] >= threshold).astype(int)

# Feature engineering: genres (multilabel)
df["genres"] = df["genres"].str.split(",")
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=mlb.classes_, index=df.index)

# Combine features
features = pd.concat([
    df[["startYear", "averageRating", "numVotes"]],
    genre_features
], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, df["isAAA"], test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Show top features
feature_importances = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=False)
print("Top predictive features:")
print(feature_importances.head(10))