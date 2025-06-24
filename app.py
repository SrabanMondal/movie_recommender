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
metadata = pd.read_parquet("data/movie_metadata.parquet")
metadata['genres'] = metadata['genres'].apply(ast.literal_eval)
unique_languages = sorted(metadata['lan'].dropna().unique().tolist())
genre_set = sorted(set(g.lower().strip() for sublist in metadata['genres'] if isinstance(sublist, list) for g in sublist))

# Helper to search movie details
def recommend_all(movies, do_content, do_user, lang, k):
    movie_names = []
    plots = []
    genres = []

    for m in movies:
        movie_names.append(m['title'])
        plots.append(m['plot'])
        genres.append(m['genres'])

    output = []

    if do_content:
        try:
            recs = recommend(movie_names, plots, genres, top_k=k, language=lang)
            output.append("🎯 Content-Based Recommendations:")
            for r in recs:
                output.append(f"• {r['title']} → {r['score']:.3f} → {r['final_score']:.2f}")
        except Exception as e:
            output.append(f"⚠️ Content-Based Error: {str(e)}")

    if do_user:
        try:
            recs = recommend_from_users(movie_names, top_k_users=10, top_k_movies=k, language=lang, provided_plots=plots, provided_genres=genres)
            output.append("👥 User-Based Recommendations:")
            for r in recs:
                output.append(f"• {r['title']} → freq {r['freq_score']} → score {r['final_score']:.2f}")
        except Exception as e:
            output.append(f"⚠️ User-Based Error: {str(e)}")

    return "\n".join(output)


def search_movies(movie_names):
    results = []
    for name in movie_names:
        row = metadata[metadata['title'].str.lower() == name.lower()]
        if not row.empty:
            row_data = row.iloc[0]
            # gr.Dataset ko list of lists chahiye
            results.append([
                name,
                row_data['plot'],
                ", ".join(row_data['genres']), # Genres ko comma-separated string banate hain
                "✅ Found"
            ])
        else:
            results.append([
                name,
                "", # User yahan plot daal sakta hai
                "", # User yahan genres daal sakta hai
                "❌ Not Found (You can add details manually)"
            ])
    return results


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Movie Recommendation App")
    gr.Markdown("Enter movies you've watched, search to autofill details, and then get recommendations.")

    with gr.Row():
        movie_input = gr.Textbox(
            label="Enter Movie Names (comma-separated)",
            placeholder="Inception, Interstellar, The Matrix",
            scale=4
        )
        search_btn = gr.Button("🔍 Search & Add Movies", scale=1)

    gr.Markdown("### Your Watched Movies (Edit plots or genres if needed)")

    # gr.Dataset is the perfect tool for this!
    movie_dataset = gr.Dataset(
        headers=["Title", "Plot", "Genres (comma-separated)", "Status"],
        datatype=["str", "str", "str", "str"],
        label="Movie Details",
        samples=None,  # Start with an empty table
        type="pandas" # It will give us a pandas DataFrame in the backend
    )

    def update_movie_list(current_df, movie_input_str):
        # Agar user ne kuch type nahi kiya to kuch mat karo
        if not movie_input_str:
            return current_df
            
        names = [m.strip() for m in movie_input_str.split(",") if m.strip()]
        
        # New search results
        search_results_list = search_movies(names)
        new_df = pd.DataFrame(search_results_list, columns=["Title", "Plot", "Genres (comma-separated)", "Status"])

        # Purane aur naye data ko merge karo
        if current_df is not None and not current_df.empty:
            # Drop duplicates, keeping the newly searched ones
            combined_df = pd.concat([current_df, new_df]).drop_duplicates(subset=['Title'], keep='last')
            return combined_df
        else:
            return new_df

    # Button click pe dataset update hoga
    search_btn.click(
        fn=update_movie_list,
        inputs=[movie_dataset, movie_input],
        outputs=[movie_dataset]
    ).then(lambda: "", outputs=movie_input) # Search ke baad input box clear kar do

    gr.Markdown("---")
    gr.Markdown("### Recommendation Settings")

    with gr.Row():
        do_content = gr.Checkbox(label="📀 Content-Based", value=True)
        do_user = gr.Checkbox(label="🤝 User-Based", value=True)
    with gr.Row():
        lang = gr.Dropdown(choices=unique_languages, label="Preferred Language", value="en")
        k = gr.Slider(minimum=5, maximum=20, step=1, label="Recommendations per method", value=10)

    submit_btn = gr.Button("🎯 Get Recommendations", variant="primary")

    output_box = gr.Textbox(label="Recommended Movies", lines=15, interactive=False)

    def collect_and_recommend(movie_df, do_content, do_user, lang, k):
        if movie_df is None or movie_df.empty:
            return "Please add at least one movie to get recommendations."

        # DataFrame se data ko list of dictionaries mein convert karo
        movies_data = []
        for _, row in movie_df.iterrows():
            # Genre string ko wapas list mein convert karo
            genres_list = [g.strip() for g in row["Genres (comma-separated)"].split(",") if g.strip()]
            
            movies_data.append({
                "title": row["Title"],
                "plot": row["Plot"],
                "genres": genres_list,
            })
        
        return recommend_all(movies_data, do_content, do_user, lang, k)

    submit_btn.click(
        fn=collect_and_recommend,
        inputs=[movie_dataset, do_content, do_user, lang, k],
        outputs=output_box
    )

if __name__=="__main__":
    demo.launch()
