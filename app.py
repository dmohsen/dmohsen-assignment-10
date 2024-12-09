from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd
import open_clip
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load CLIP model and preprocessing tools
clip_model, _, image_preprocessor = open_clip.create_model_and_transforms(
    model_name='ViT-B/32', pretrained='openai', cache_dir='cache'
)
text_tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_model.eval()

# Device configuration (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# Load precomputed image embeddings
embeddings_db = pd.read_pickle('static/images/image_embeddings.pickle')

# Utility function: Retrieve top-k results by cosine similarity
def find_similar_images(query_embedding, num_results=5):
    # Compute cosine similarities
    similarities = embeddings_db['embedding'].apply(
        lambda stored_embedding: F.cosine_similarity(
            torch.tensor(query_embedding, device=device),
            torch.tensor(stored_embedding, device=device)
        ).item()
    )

    # Retrieve indices of top-k results
    best_matches = similarities.nlargest(num_results).index

    # Prepare results list with file names and similarity scores
    results = [
        {
            "file_name": f"coco_images_resized/{embeddings_db.loc[idx, 'file_name']}",
            "similarity": round(similarities[idx], 3)
        }
        for idx in best_matches
    ]
    return results

@app.route("/")
def home():
    """Serve the main search interface."""
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def perform_search():
    """Handle search requests and return results."""
    query_type = request.form.get("query_type")
    num_results = int(request.form.get("top_k", 5))

    query_embedding = None

    # Validate and handle the query type
    if query_type == "text":
        user_text = request.form.get("text_query")
        tokenized_text = text_tokenizer([user_text]).to(device)
        query_embedding = F.normalize(clip_model.encode_text(tokenized_text)).squeeze(0).cpu().numpy()

    elif query_type == "image":
        uploaded_image = request.files["image_query"]
        processed_image = image_preprocessor(Image.open(uploaded_image).convert("RGB")).unsqueeze(0).to(device)
        query_embedding = F.normalize(clip_model.encode_image(processed_image)).squeeze(0).cpu().numpy()

    elif query_type == "combined":
        user_text = request.form.get("text_query")
        weight = float(request.form.get("lambda", 0.5))

        # Process image query
        uploaded_image = request.files["image_query"]
        processed_image = image_preprocessor(Image.open(uploaded_image).convert("RGB")).unsqueeze(0).to(device)
        image_embedding = F.normalize(clip_model.encode_image(processed_image))

        # Process text query
        tokenized_text = text_tokenizer([user_text]).to(device)
        text_embedding = F.normalize(clip_model.encode_text(tokenized_text))

        # Combine text and image embeddings
        query_embedding = F.normalize(weight * text_embedding + (1 - weight) * image_embedding).squeeze(0).cpu().numpy()

    else:
        return jsonify({"error": "Invalid query type provided"}), 400

    # Retrieve and return top-k search results
    search_results = find_similar_images(query_embedding, num_results)
    return jsonify(search_results)

@app.route('/coco_images_resized/<filename>')
def serve_coco_image(filename):
    """Serve images from the coco_images_resized directory."""
    return send_from_directory('static/images/coco_images_resized', filename)

if __name__ == "__main__":
    app.run(debug=True)
