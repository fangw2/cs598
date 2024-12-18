import streamlit as st
import pandas as pd
import numpy as np
import os

Rmat = pd.read_csv("Rmat_small.csv", index_col=0)
S_optimized = pd.read_csv("S_optimized.csv", index_col=0)
popularity_ranking = pd.read_csv("popularity_ranking.csv")

movies = pd.read_csv("movies.dat", sep="::", engine="python", header=None,
                     names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1")

if popularity_ranking['MovieID'].dtype != object:
    popularity_ranking['MovieID'] = popularity_ranking['MovieID'].apply(lambda x: f"m{x}")

title_map = dict(zip(movies['MovieID'], movies['Title']))

selected_movies = Rmat.columns[:100]
Rmat_small = Rmat[selected_movies]
S_small = S_optimized.loc[selected_movies, selected_movies]

def myIBCF(newuser, S, Rmat, popularity_ranking):
    user_ratings = pd.Series(newuser, index=Rmat.columns)
    predictions = []
    for i, movie_id in enumerate(Rmat.columns):
        if np.isnan(user_ratings[movie_id]):
            sim_vector = S.loc[movie_id, :]
            rated_mask = ~user_ratings.isna()
            neighbor_mask = ~sim_vector.isna()
            valid_mask = rated_mask & neighbor_mask
            valid_neighbors = sim_vector[valid_mask]

            if len(valid_neighbors) > 0:
                weighted_sum = np.sum(valid_neighbors.values * user_ratings[valid_neighbors.index].values)
                weight_sum = np.sum(valid_neighbors.values)
                predicted_rating = weighted_sum / weight_sum if weight_sum != 0 else np.nan
            else:
                predicted_rating = np.nan
        else:
            predicted_rating = np.nan

        predictions.append(predicted_rating)

    preds_df = pd.DataFrame({'MovieID': Rmat.columns, 'Prediction': predictions})
    already_rated = user_ratings[~user_ratings.isna()].index
    preds_df = preds_df[~preds_df['MovieID'].isin(already_rated)]

    preds_sorted = preds_df.sort_values('Prediction', ascending=False)
    top_predictions = preds_sorted.dropna(subset=['Prediction']).head(10)

    if len(top_predictions) < 10:
        needed = 10 - len(top_predictions)
        chosen = set(top_predictions['MovieID']) | set(already_rated)
        fallback_candidates = popularity_ranking[popularity_ranking['MovieID'].isin(Rmat.columns)]
        fallback_candidates = fallback_candidates[~fallback_candidates['MovieID'].isin(chosen)]
        fallback = fallback_candidates.head(needed)
        fallback['Prediction'] = np.nan
        top_predictions = pd.concat([top_predictions, fallback[['MovieID', 'Prediction']]], ignore_index=True)

    return top_predictions.head(10)

st.title("System II: Movie Recommendation App")

st.markdown("""
**Step 1: Rate Movies**  
Select a rating (1-5 stars) for as many of the following 100 movies as possible. 
If you don't want to rate a movie, choose "Not Rated".

**Step 2: Discover Movies You Might Like**  
After rating, click "Get Recommendations" to see your top 10 recommended movies, along with their posters.
""")

st.markdown("---")

st.header("Step 1: Rate as Many Movies as Possible")

star_options = [
    "Not Rated",
    "★☆☆☆☆ (1 star)",
    "★★☆☆☆ (2 stars)",
    "★★★☆☆ (3 stars)",
    "★★★★☆ (4 stars)",
    "★★★★★ (5 stars)"
]

user_ratings = []
for movie_id in selected_movies:
    # Extract numeric movie ID for title and poster lookup
    numeric_id = int(movie_id[1:])
    title = title_map.get(numeric_id, movie_id)  # fallback to movie_id if no title found
    image_path = os.path.join("Selected100", f"{numeric_id}.jpg")

    # We'll use columns to arrange poster and title on one row, and the rating selectbox below
    col1, col2 = st.columns([1, 4])
    with col1:
        if os.path.exists(image_path):
            st.image(image_path, width=80)
        else:
            st.write("No poster available.")

    with col2:
        st.write(f"**{title}**")

    rating_str = st.selectbox(f"Rate {title}", star_options, index=0)
    if rating_str == "Not Rated":
        user_ratings.append(np.nan)
    else:
        rating_val = star_options.index(rating_str)  # returns an integer 1-5
        user_ratings.append(float(rating_val))

st.markdown("---")


st.header("Step 2: Discover Movies You Might Like")

if st.button("Get Recommendations"):
    user_ratings_array = np.array(user_ratings)
    recommendations = myIBCF(user_ratings_array, S_small, Rmat_small, popularity_ranking)
    
    st.subheader("Your Top 10 Recommendations:")
    if len(recommendations) == 0:
        st.write("No recommendations found. Try rating more movies!")
    else:
        for idx, row in recommendations.iterrows():
            movie_id = row['MovieID']
            pred_rating = row['Prediction']
            pred_str = f"{pred_rating:.2f}" if not np.isnan(pred_rating) else "N/A"

            numeric_id = int(movie_id[1:])
            title = title_map.get(numeric_id, movie_id)  # fallback to movie_id if title not found

            image_path = os.path.join("Selected100", f"{numeric_id}.jpg")

            col1, col2 = st.columns([2,1])
            with col1:
                st.write(f"**{title}** (Predicted Rating: {pred_str})")
            with col2:
                if os.path.exists(image_path):
                    st.image(image_path, width=100)
                else:
                    st.write("No poster available.")