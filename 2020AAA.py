import pandas as pd

# Paths to your downloaded datasets
title_basics_path = "title.basics.tsv.gz"
title_ratings_path = "title.ratings.tsv.gz"

# Load the basics and ratings data
title_basics = pd.read_csv(title_basics_path, sep="\t", compression="gzip", dtype=str, na_values="\\N")
title_ratings = pd.read_csv(title_ratings_path, sep="\t", compression="gzip", dtype={"tconst": str, "averageRating": float, "numVotes": int})

# Filter for 2020 movies
movies_2020 = title_basics[(title_basics["titleType"] == "movie") & (title_basics["startYear"] == "2020")]

# Merge with ratings
movies_2020_rated = pd.merge(movies_2020, title_ratings, on="tconst")

### Filter for movies with at least 1000 votes ####
movies_2020_rated = movies_2020_rated[movies_2020_rated["numVotes"] >= 1000]

# Get the top-rated movie in 2020 (balanced by numVotes)
top_movie = movies_2020_rated.sort_values(by=["averageRating", "numVotes"], ascending=[False, False]).head(1)

# Display the result
print(top_movie[["primaryTitle", "averageRating", "numVotes", "startYear"]])