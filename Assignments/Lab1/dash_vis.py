import dash
from dash import html, dcc, Input, Output
from data_processing import get_df
from feature_engineering import create_tfidf_matrix, filter_out_suspect_users, nearest_neighbors_model
from recommender import get_recommendations_nn, get_recommendations_ratings, get_title_rating


# ------ Data Preparation ------

df = get_df()
cleaned_df, mean_ratings = filter_out_suspect_users(df)
tfidf_matrix, movie_df = create_tfidf_matrix(cleaned_df)
nn = nearest_neighbors_model(tfidf_matrix, n_neighbors=50)
movie_titles = movie_df["title"].str.lower().unique()

# ------ Dash App Layout ------

app = dash.Dash(__name__)
app.title = "Movie Recommender"
app.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f4f6f9',
        'minHeight': '100vh',
        'padding': '20px'
    },
    children=[
        html.H1("Movie Recommender", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '30px'
                    }
                ),
        
        html.Div([
            dcc.Dropdown(
                id='movie-input',
                options=[{'label': title.title(), 'value': title} for title in movie_titles],
                placeholder='Select a movie...',
                style={'width': '70%', 'marginRight': '10px'}
            ),
        ], style={
            'textAlign': 'center', 
            'marginBottom': '30px'
            }),

        html.Div(id='title-rating', style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }),

        html.Div([
            
            html.Div([
                html.H3("Top 5 Highest Rated Recommendations", style={'color': '#2c3e50'}),
                html.Ul(id='highest-rated-list')
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                'width': '45%'
            }),

            html.Div([
                html.H3("Top 5 Similarily Rated Recommendations", style={'color': '#2c3e50'}),
                html.Ul(id='similarily-rated-list')
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                'width': '45%'
            }),
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'gap': '20px'
        })
    ]
)

# ------ Callbacks ------
@app.callback(
    Output('title-rating', 'children'),
    Output('highest-rated-list', 'children'),
    Output('similarily-rated-list', 'children'),
    Input('movie-input', 'value')
)
def update_recommendations(movie_title):
    if not movie_title:
        return [], [], []
    
    movie_title = movie_title.title()

    recs = get_recommendations_nn(movie_title, movie_df, tfidf_matrix, nn, top_n=50)
    highest_rated, similarily_rated = get_recommendations_ratings(mean_ratings, recs, movie_title)
    title_rating = get_title_rating(mean_ratings, movie_title)

    card = html.Div([
        html.H2(movie_title, style={'marginBottom': '10px'}),
        html.P(f"Average Rating: {title_rating:.2f}")
    ])

    return card, \
           [html.Li(f"{title} - rating: {rating:.2f}") for title, rating in highest_rated], \
           [html.Li(f"{title} - rating: {rating:.2f}") for title, rating in similarily_rated]

    
# ------ Run App ------

if __name__ == '__main__':
    app.run(debug=True)


'''

 "Skapa en layout för en Dash-app där användaren kan söka efter något via en dropdown-sökfält. 
    Layouten ska ha en huvudrubrik centrerad högst upp, en dropdown centrerad under rubriken, 
    en ruta under dropdownen som visar statistik eller information om det valda objektet och 
    två rutor bredvid varandra där rekommendationer visas som listor."

ChatGPT:
import dash
from dash import html, dcc

app = dash.Dash(__name__)
app.title = "Title"

app.layout = html.Div(
    style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#f4f6f9',
        'minHeight': '100vh',
        'padding': '20px'
    },
    children=[
        # Header
        html.H1("Projektets Dashboard", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '30px'
                }
        ),
        
        # Dropdown-sökfält
        html.Div([
            dcc.Dropdown(
                id='input-dropdown',
                options=[],  # Tom, kan fyllas senare
                placeholder='Välj något...',
                style={'width': '70%', 'marginRight': '10px'}
            ),
        ], style={
            'textAlign': 'center', 
            'marginBottom': '30px'
        }),

        # Statistikruta
        html.Div(id='stats-output', style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '12px',
            'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'minHeight': '100px'
        }),

        # Två rekommendationsrutor
        html.Div([
            html.Div([
                html.H3("Rekommendation 1", style={'color': '#2c3e50'}),
                html.Ul(id='rec1-list', style={'listStyleType': 'none', 'padding': 0})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                'width': '100%',
                'maxWidth': '45%',
                'marginBottom': '20px'
            }),

            html.Div([
                html.H3("Rekommendation 2", style={'color': '#2c3e50'}),
                html.Ul(id='rec2-list', style={'listStyleType': 'none', 'padding': 0})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 10px rgba(0,0,0,0.1)',
                'width': '100%',
                'maxWidth': '45%',
                'marginBottom': '20px'
            }),
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'gap': '20px',
            'flexWrap': 'wrap'
        }),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
'''