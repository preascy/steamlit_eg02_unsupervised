import streamlit as st
import pandas as pd
import numpy as np

# Sample data for the recommender system
movie_data = {
    'MovieID': [1, 2, 3, 4, 5],
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Action', 'Drama', 'Comedy']
}

ratings_data = {
    'UserID': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    'MovieID': [1, 2, 3, 2, 4, 1, 5, 3, 4, 5],
    'Rating': [5, 4, 3, 5, 3, 4, 5, 2, 4, 3]
}

movies = pd.DataFrame(movie_data)
ratings = pd.DataFrame(ratings_data)

def recommend_movies(user_id, ratings, movies, num_recommendations=3):
    # Get the movies rated by the user
    user_ratings = ratings[ratings['UserID'] == user_id]
    rated_movie_ids = user_ratings['MovieID'].tolist()
    
    # Recommend movies not rated by the user
    recommended_movies = movies[~movies['MovieID'].isin(rated_movie_ids)]
    recommendations = recommended_movies.sample(num_recommendations)
    
    return recommendations

# Streamlit app with pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Us", "Recommendation"])

if page == "Home":
    st.title("Welcome to the Movie Recommender System")
    st.write("""
    This application helps you find movies that you'll love based on your preferences and ratings.
    
    Navigate to the Recommendation page to get personalized movie recommendations.
    """)

elif page == "About Us":
    st.title("About Us")
    st.write("""
    Welcome to the Simple Movie Recommender System!

    Our mission is to help you find movies that you'll love based on your preferences and ratings.
    
    This system uses collaborative filtering to suggest movies you haven't seen yet but might enjoy based on the ratings of other users with similar tastes.

    **Team Members:**
    - Neo Modibedi: Team Leader
    - Melody Msimango: Project Manager
    - Priscilla Matuludi: GitHub Manager
    - Keryn Rabe: Organizer
    - Tumelo Matamane
    - Ntandoyenkosi Biyela

    Enjoy your movie recommendations!
    """)

elif page == "Recommendation":
    st.title("Movie Recommendations")
    
    # User input for UserID
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=5, value=1)

    if st.button("Recommend Movies"):
        recommendations = recommend_movies(user_id, ratings, movies)
        st.write("Recommended Movies for User ID {}:".format(user_id))
        st.table(recommendations[['Title', 'Genre']])


