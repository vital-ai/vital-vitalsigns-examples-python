import pickle
import tensorflow as tf
import tf2onnx
from chars2vec import Chars2Vec


def load_model(path):

    path_to_model = path

    with open(path_to_model + '/model.pkl', 'rb') as f:
        structure = pickle.load(f)
        emb_dim, char_to_ix = structure[0], structure[1]

    c2v_model = Chars2Vec(emb_dim, char_to_ix)
    c2v_model.embedding_model.load_weights(path_to_model + '/weights.h5')
    c2v_model.embedding_model.compile(optimizer='adam', loss='mae')

    return c2v_model


def load_and_save_model_as_savedmodel(pickle_path, weights_path, saved_model_dir):
    with open(pickle_path, 'rb') as f:
        emb_dim, char_to_ix = pickle.load(f)[:2]

    # Rebuild the model architecture with precise layer names
    lstm_input = tf.keras.layers.Input(shape=(None, len(char_to_ix)), name="lstm_input")

    x = tf.keras.layers.LSTM(emb_dim, return_sequences=True, name="lstm_1")(lstm_input)
    x = tf.keras.layers.LSTM(emb_dim, name="lstm_2")(x)

    embedding_model = tf.keras.models.Model(inputs=[lstm_input], outputs=x, name="embedding_model")

    model_input_1 = tf.keras.layers.Input(shape=(None, len(char_to_ix)), name="model_input_1")
    model_input_2 = tf.keras.layers.Input(shape=(None, len(char_to_ix)), name="model_input_2")

    embedding_1 = embedding_model(model_input_1)
    embedding_2 = embedding_model(model_input_2)
    x = tf.keras.layers.Subtract(name="subtract")([embedding_1, embedding_2])
    x = tf.keras.layers.Dot(axes=1, name="dot")([x, x])
    model_output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)

    model = tf.keras.models.Model(inputs=[model_input_1, model_input_2], outputs=model_output, name="chars2vec_model")
    model.compile(optimizer='adam', loss='mae')

    # Load weights
    model.load_weights(weights_path)

    # Save the complete model
    model.save(saved_model_dir)


def convert_embedding_model_to_onnx(pickle_path, weights_path, onnx_model_path):

    with open(pickle_path, 'rb') as f:
        emb_dim, char_to_ix = pickle.load(f)[:2]

    # Rebuild the embedding model architecture
    lstm_input = tf.keras.layers.Input(shape=(None, len(char_to_ix)), name="lstm_input")

    x = tf.keras.layers.LSTM(emb_dim, return_sequences=True, name="lstm_1")(lstm_input)
    x = tf.keras.layers.LSTM(emb_dim, name="lstm_2")(x)

    embedding_model = tf.keras.models.Model(inputs=[lstm_input], outputs=x, name="embedding_model")

    # Load weights
    embedding_model.load_weights(weights_path, by_name=True)

    # Convert to ONNX
    spec = (tf.TensorSpec((None, None, len(char_to_ix)), tf.float32, name="lstm_input"),)
    model_proto, _ = tf2onnx.convert.from_keras(embedding_model, input_signature=spec, output_path=onnx_model_path)
    return model_proto


def main():

    print('Convert Chars2Vec Model')

    models_dir = "/Users/hadfield/Local/external-git/chars2vec/chars2vec/trained_models"

    model_dir = f"{models_dir}/eng_300/"

    pickle_path = f"{models_dir}/eng_300/model.pkl"
    weights_path = f"{models_dir}/eng_300/weights.h5"

    onnx_model_path = '../en_300_model/model.onnx'

    model = load_model(model_dir)

    # convert_to_onnx(model.model, onnx_model_path)

    convert_embedding_model_to_onnx(pickle_path, weights_path, onnx_model_path)

    # onnx_model_path = 'chars2vec_en_300.onnx'

    # load_and_save_model_as_savedmodel(pickle_path, weights_path, complete_model_path)

    # convert_to_onnx(model_dir, 'chars2vec_en_300.onnx')


if __name__ == "__main__":
    main()
