import pandas as pd

# Load the IMDb title basics file
title_basics = pd.read_csv(
    "title.basics.tsv.gz",
    sep='\t',
    compression='gzip',
    dtype=str,  # to avoid dtype issues
    na_values='\\N'  # handle missing values properly
)

# Load the IMDb ratings file
title_ratings = pd.read_csv(
    "title.ratings.tsv.gz",
    sep='\t',
    compression='gzip',
    dtype={'tconst': str, 'averageRating': float, 'numVotes': int}
)

# Load the IMDb name basics file
name_basics = pd.read_csv(
    "name.basics.tsv.gz",
    sep='\t',
    compression='gzip',
    dtype=str,
    na_values='\\N'
)

# Preview the datasets
print("Title Basics:")
print(title_basics.head())

print("\nTitle Ratings:")
print(title_ratings.head())

print("\nName Basics:")
print(name_basics.head())