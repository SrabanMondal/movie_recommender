import gradio as gr
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import ast
load_dotenv()
hf_token = os.getenv("HF_TOKEN") 
if hf_token is None:
    raise ValueError("HF_TOKEN not found in environment variables")

repo_id = "srabanmondal/moviedata"
parent_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(parent_dir, "data")
os.mkdir(data_dir)
local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",            
    token=hf_token,
    local_dir=data_dir,
    local_dir_use_symlinks=False    
)
print("Downloaded to:", local_dir)
from utils.recommend import recommend, recommend_from_users
metadata = pd.read_parquet(local_dir+"/movie_metadata.parquet")
unique_languages = sorted(metadata['lan'].dropna().unique().tolist())
genre_set = sorted(set(g.lower().strip() for sublist in metadata['genres'] if isinstance(sublist, list) for g in sublist))



def search_movie(title, names, plots, genres):
    title_clean = title.strip().lower()
    
    # Case-insensitive match
    match = metadata[metadata['title'].str.strip().str.lower() == title_clean]

    if not match.empty:
        row = match.iloc[0]
        names.append(row['title'].strip())
        plots.append(row['plot'].strip())

        # Ensure genres is a list
        genre_list = row['genres']
        if isinstance(genre_list, str):
            import ast
            genre_list = ast.literal_eval(genre_list)

        genres.append(genre_list)
        status = f"✅ Found '{row['title']}' and added."
    else:
        status = f"❌ '{title}' not found. Please add manually."

    return "", names, plots, genres, status

# Function to add movie to 3 separate lists
def add_movie(name, plot, genres_text, names, plots, genres):
    if name.strip() and plot.strip():
        genre_list = [g.strip() for g in genres_text.split(',') if g.strip()]
        names.append(name.strip())
        plots.append(plot.strip())
        genres.append(genre_list)
    return "", "", "", names, plots, genres

# Function to clear all states
def clear_all():
    return [], [], []

# Function to make structured JSON for display
def make_structured_json(names, plots, genres):
    return [
        {"title": n, "plot": p, "genres": g}
        for n, p, g in zip(names, plots, genres)
    ]

def recommend_func(movie_names, plots, genres, top_n, content_based, user_based, language):
    content_df = pd.DataFrame()
    user_df = pd.DataFrame()
    if content_based:
        content_df = recommend(movie_names,plots,genres,top_n, language)
    if user_based:
        user_df = recommend_from_users(movie_names,10,top_n,language,plots,genres)
    return content_df, user_df

with gr.Blocks() as demo:
    movie_names_state = gr.State([])
    plots_state = gr.State([])
    genres_state = gr.State([])

    gr.Markdown("## 🎬 Add Movies with Title, Plot, and Genres")
    with gr.Row():
      search_input = gr.Textbox(label="🔍 Search Movie Title")
      search_button = gr.Button("Search & Add")

    status_display = gr.Markdown("")  # For search status
    with gr.Row():
        name_input = gr.Textbox(label="🎞️ Movie Name")
        genres_input = gr.Textbox(label="🏷️ Genres (comma-separated)")

    plot_input = gr.Textbox(lines=4, label="📝 Plot (multiline allowed)")

    add_button = gr.Button("➕ Add Movie")
    clear_button = gr.Button("🗑️ Clear All")

    # Final structured format only
    structured_display = gr.JSON(label="📦 Structured Movie Data")
    search_button.click(
      fn=search_movie,
      inputs=[search_input, movie_names_state, plots_state, genres_state],
      outputs=[search_input, movie_names_state, plots_state, genres_state, status_display]
    ).then(
      fn=make_structured_json,
      inputs=[movie_names_state, plots_state, genres_state],
      outputs=structured_display
    )

    # Add movie
    add_button.click(
        fn=add_movie,
        inputs=[name_input, plot_input, genres_input, movie_names_state, plots_state, genres_state],
        outputs=[name_input, plot_input, genres_input, movie_names_state, plots_state, genres_state]
    ).then(
        fn=make_structured_json,
        inputs=[movie_names_state, plots_state, genres_state],
        outputs=structured_display
    )

    # Clear all
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[movie_names_state, plots_state, genres_state]
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=structured_display
    )
    top_n_slider = gr.Slider(5, 20, step=1, value=10, label="🔢 Top N Results")
    with gr.Row():
      content_checkbox = gr.Checkbox(label="📚 Content-Based")
      user_checkbox = gr.Checkbox(label="👤 User-Based")
    language_selector = gr.Dropdown(
      choices=unique_languages,
      label="🌐 Select Language",
      value="en"  # default value
    )

    generate_button = gr.Button("🔮 Generate Output")
    content_output_box = gr.Dataframe(
        label="📚 Content-Based Recommendations",
        headers=["title", "genres", "final_score"],
        interactive=False
    )

    user_output_box = gr.Dataframe(
        label="👤 User-Based Recommendations",
        headers=["title", "genres", "final_score"],
        interactive=False
    )


    generate_button.click(
        fn=recommend_func,
        inputs=[
        movie_names_state, 
        plots_state, 
        genres_state,
        top_n_slider,
        content_checkbox,
        user_checkbox,
        language_selector
    ],
        outputs=[
        content_output_box,
        user_output_box
    ]

    )

if __name__=="__main__":
    demo.launch()
