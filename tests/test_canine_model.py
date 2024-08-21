from scipy.spatial.distance import cdist
from transformers import CanineTokenizer, CanineModel
# import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Load the CANINE model and tokenizer
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    model = CanineModel.from_pretrained("google/canine-c")

    # Encode the strings
    strings = ["john smithe", "mstih ojnh"]

    strings = ["Bob Jones", "bob jones"]

    # strings = ["john smithe", "john smith"]

    inputs = tokenizer(strings, padding="longest", truncation=True, return_tensors="pt")

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state

    # Average pooling to get a single vector per string
    embeddings = embeddings.mean(dim=1).detach().numpy()

    print(embeddings)

    print(f"Vector Size: {embeddings.shape[1]}")

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    print(f"Cosine Similarity: {similarity[0][0]}")

    distance = cdist([embeddings[0]], [embeddings[1]], metric='cosine')[0][0]
    print(f"Cosine Distance: {distance}")


if __name__ == "__main__":
    main()
