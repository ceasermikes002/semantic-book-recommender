# Book Recommender System

This project is a semantic book recommendation system using Python, Gradio, and modern NLP/ML tools. It provides book recommendations based on user queries, category, and emotional tone, leveraging vector search and emotion classification.

## Features
- **Semantic Search**: Uses vector embeddings for book descriptions to find relevant recommendations.
- **Emotion & Category Filtering**: Classifies books by emotion and category for more personalized results.
- **Interactive Dashboard**: Gradio-based web UI for easy interaction.
- **Data Exploration**: Jupyter notebooks for data cleaning, classification, and analysis.

## Project Structure
- `gradio-dashboard.py`: Main app/dashboard code.
- `data-exploration.ipynb`, `text_classification.ipynb`, `sentiment_analysis.ipynb`, `vector-search.ipynb`: Notebooks for data processing and model development.
- `books_cleaned.csv`, `books_with_categories.csv`, `books_with_emotions.csv`: Processed datasets.
- `tagged_description.txt`: Text file for vector search.
- `cover-not-found.jpg`: Placeholder image for missing book covers.

## Quickstart (Local)
1. **Clone the repository** and enter the project folder.
2. **Install dependencies** (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install gradio pandas numpy langchain chromadb langchain-google-genai python-dotenv transformers tqdm seaborn matplotlib
   ```
3. **Set up environment variables**:
   - Create a `.env` file with your Google Generative AI API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```
4. **Run the dashboard**:
   ```bash
   python gradio-dashboard.py
   ```
   The app will launch in your browser at `http://localhost:7860`.

## Deploying Live (Cheapest & Easiest)
### Option 1: [Hugging Face Spaces](https://huggingface.co/spaces) *(Recommended)*
1. **Create a Hugging Face account** and a new Space (select Gradio as the SDK).
2. **Upload your code and data files** (including `gradio-dashboard.py`, CSVs, and `cover-not-found.jpg`).
3. **Add your environment variables** in the Space settings ("Secrets") for your API keys.
4. **Add a `requirements.txt`** with all dependencies (see above).
5. **Deploy** — Hugging Face will build and host your app for free (with some resource limits).

### Option 2: [Render.com](https://render.com/)
1. **Create a free account** and a new "Web Service".
2. **Connect your GitHub repo** or upload your code.
3. **Set build command**: `pip install -r requirements.txt`
4. **Set start command**: `python gradio-dashboard.py`
5. **Add environment variables** in the dashboard.
6. **Deploy** — Free tier available, easy setup.

### Option 3: [Replit](https://replit.com/)
1. **Create a new Python Repl**.
2. **Upload your code and data files**.
3. **Add environment variables** in the Secrets tab.
4. **Install dependencies** via the "Packages" tool or `requirements.txt`.
5. **Run the app** — Replit provides a public URL.

## Notes
- For best results, use Hugging Face Spaces for free, easy, and fast deployment.
- Make sure your API keys are kept secret (use environment variables, not hardcoding).
- For large datasets, you may need to reduce file size or use cloud storage.

## License
MIT License

---

*For questions or improvements, open an issue or pull request!*
