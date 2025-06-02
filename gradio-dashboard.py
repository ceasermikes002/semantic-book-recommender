import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    candidate_recs = books[books["isbn13"].isin(books_list)].copy()

    if category and category != 'All':
        candidate_recs = candidate_recs[candidate_recs["simple_categories"] == category]

    if tone == "Happy":
        candidate_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == "Surprising":
        candidate_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == "Angry":
        candidate_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == "Suspenseful":
        candidate_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == "Sad":
        candidate_recs.sort_values(by='sadness', ascending=False, inplace=True)

    final_recommendations = candidate_recs.head(final_top_k)

    return final_recommendations


def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncates_desc_split = description.split()
        truncated_description = " ".join(truncates_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_joined = ", ".join(authors_split[:-1])
            authors_str = f"{authors_joined} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append([row['large_thumbnail'], caption])

    return results


valid_categories = books['simple_categories'].dropna().astype(str).unique().tolist()
categories = ["All"] + sorted(valid_categories)

tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Book Recommendation System")

    # This row contains all input fields and the submit button
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book you would like to read",
                                placeholder="A story of betrayal", scale=2)  # Added scale for wider textbox
        category_dropdown = gr.Dropdown(choices=categories, label="Choose a category", value="All", scale=1)
        tone_dropdown = gr.Dropdown(choices=tones, label="Choose a tone", value="All", scale=1)
        submit_button = gr.Button("Find recommendations", scale=0)  # Smaller button, fits row

    # This column will contain the output, spanning the full width below the inputs
    with gr.Column():
        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended books", columns=8, rows=2, object_fit="contain", height="auto")
        # object_fit="contain" helps images fit within their space
        # height="auto" allows the gallery to adjust height dynamically

    submit_button.click(recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=[output])

if __name__ == '__main__':
    dashboard.launch()