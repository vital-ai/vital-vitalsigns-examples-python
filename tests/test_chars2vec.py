import chars2vec
import hnswlib
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd


def main():
    # words = ['Natural', 'Language', 'Understanding',
    #         'Naturael', 'Longuge', 'Updderctundjing',
    #         'Motural', 'Lamnguoge', 'Understaating',
    #         'Naturrow', 'Laguage', 'Unddertandink',
    #         'Nattural', 'Languagge', 'Umderstoneding']

    # Models names: 'eng_50', 'eng_100', 'eng_150', 'eng_200', 'eng_300'
    c2v_model = chars2vec.load_model('eng_300')

    words = ['John', 'Jon', 'Bob', 'Frederick', 'Frederik', 'Marc Hadfield']

    # Create word embeddings
    word_embeddings = c2v_model.vectorize_words(words)

    dim = word_embeddings.shape[1]  # Dimension of the embeddings

    num_elements = len(words)

    p = hnswlib.Index(space='cosine', dim=dim)  # cosine distance metric
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Add vectors to the index
    p.add_items(word_embeddings, np.arange(num_elements))

    # Set query parameters
    p.set_ef(50)  # ef should always be greater than k

    # print(word_embeddings)

    def find_closest_string(query_string, model, index, words_list):
        # Convert the query string to a vector
        query_vector = model.vectorize_words([query_string])

        # Perform the search
        labels, distances = index.knn_query(query_vector, k=1)

        # Get the closest string and the distance
        closest_string = words_list[labels[0][0]]
        closest_distance = distances[0][0]

        return closest_string, closest_distance

    distances = cdist(word_embeddings, word_embeddings, metric='cosine')

    word_pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word_pairs.append((words[i], words[j], distances[i, j]))

    # Convert to DataFrame for better visualization
    distance_df = pd.DataFrame(word_pairs, columns=['Word 1', 'Word 2', 'Distance'])

    # Display the DataFrame
    print(distance_df.to_string(index=False))

    query = "Mark Hatfield"
    closest_word, distance = find_closest_string(query, c2v_model, p, words)
    print(f'The closest word to "{query}" is "{closest_word}" with a distance of {distance:.4f}')


if __name__ == "__main__":
    main()
