import numpy as np
import logging
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, target_vocab_size=2**13, max_length=100):
        self.target_vocab_size = target_vocab_size
        self.max_length = max_length
        self.tokenizer = None

    def clean_text(self, texts):
        cleaned_data = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text) 
            text = re.sub(r'\s+', ' ', text).strip() 
            cleaned_data.append(text)
        return cleaned_data

    def build_tokenizer(self, corpus):

        logger.info("Building tokenizer...")
        corpus_filtered = [s for s in corpus if isinstance(s, str)]
        if not corpus_filtered:
            logger.warning("Corpus is empty or contains no strings. Tokenizer might not be effective.")
            self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                ["<pad>", "<unk>"], target_vocab_size=self.target_vocab_size
            )
        else:
            self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                corpus_filtered, target_vocab_size=self.target_vocab_size
            )
        logger.info(f"Vocabulary size: {self.tokenizer.vocab_size}")
        return self.tokenizer

    def encode_texts(self, texts):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not built. Call build_tokenizer first.")

        token_ids = [self.tokenizer.encode(text) for text in texts]
        token_ids = pad_sequences(token_ids, maxlen=self.max_length, padding='post')
        return token_ids

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        self.wq = layers.SpectralNormalization(layers.Dense(d_model, name='query'))
        self.wk = layers.SpectralNormalization(layers.Dense(d_model, name='key'))
        self.wv = layers.SpectralNormalization(layers.Dense(d_model, name='value'))
        self.dense = layers.SpectralNormalization(layers.Dense(d_model, name='output'))

    def split_heads(self, x, batch_size):
    
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9) # Add a large negative number to masked positions

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q, mask=None):

        batch_size = tf.shape(q)[0]

        # Project queries, keys, and values
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calculate scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # Transpose and concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # Final dense layer
        output = self.dense(concat_attention)
        return output, attention_weights

class TextGenerator:
   
    def __init__(self, sequence_length, embedding_dim, d_model=None, num_heads=None): # d_model and num_heads are not directly used in LSTM layers, but for MHA
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.d_model = d_model # Stored for MHA
        self.num_heads = num_heads # Stored for MHA

    def build_model(self):

        inputs = layers.Input(shape=(self.sequence_length, self.embedding_dim),
                              name='generator_input')

        x = layers.Bidirectional(
            layers.LSTM(self.d_model // 2, return_sequences=True), # Use d_model for LSTM units
            name='bilstm_1'
        )(inputs)

        x = layers.Bidirectional(
            layers.LSTM(self.d_model // 2, return_sequences=True), # Output shape (batch, seq_len, d_model)
            name='bilstm_2'
        )(x)

        attention_output, _ = MultiHeadAttention(
            self.d_model, self.num_heads
        )(x, x, x) # Query, Key, Value are all from the LSTM output

        x = layers.Concatenate(name='concat_attention')([x, attention_output])

        x = layers.TimeDistributed(
            layers.SpectralNormalization(layers.Dense(self.embedding_dim, activation='tanh')),
            name='output_projection'
        )(x)

        model = models.Model(inputs, x, name="Generator")
        return model

class TextDiscriminator:

    def __init__(self, sequence_length, embedding_dim, d_model, num_heads):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads

    def build_model(self):
        inputs = layers.Input(shape=(self.sequence_length, self.embedding_dim),
                              name='discriminator_input')

        x = layers.SpectralNormalization(layers.Conv1D(64, 3, activation='relu', padding='same',
                                                      name='conv1d_1'))(inputs)
        x = layers.SpectralNormalization(layers.Conv1D(32, 3, activation='relu', padding='same',
                                                      name='conv1d_2'))(x)

        if x.shape[-1] != self.d_model:
            x_proj = layers.SpectralNormalization(layers.Dense(self.d_model, activation='relu', name='attention_input_projection'))(x)
        else:
            x_proj = x

        attention_output, _ = MultiHeadAttention(
            self.d_model, self.num_heads, name='discriminator_attention'
        )(x_proj, x_proj, x_proj) # Q, K, V from projected conv output

        if x.shape[-1] != attention_output.shape[-1]:
            x_to_concat = layers.SpectralNormalization(layers.Dense(attention_output.shape[-1], activation='relu', name='conv_output_for_concat'))(x)
        else:
            x_to_concat = x

        x = layers.Concatenate(name='discriminator_concat_attention')([x_to_concat, attention_output])


        x = layers.LSTM(64, return_sequences=True, name='lstm_1')(x)
        x = layers.LSTM(32, name='lstm_2')(x) # Return only the last output for classification

        x = layers.SpectralNormalization(layers.Dense(64, activation='relu', name='dense_1'))(x)
        x = layers.Dropout(0.3, name='dropout')(x) # Dropout for regularization
        x = layers.SpectralNormalization(layers.Dense(1, activation='sigmoid', name='output'))(x) # Sigmoid for binary classification

        model = models.Model(inputs, x, name="Discriminator")
        return model

class SentenceGAN:

    def __init__(self, sequence_length, embedding_dim, d_model, num_heads):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.d_model = d_model # Pass d_model to discriminator
        self.num_heads = num_heads # Pass num_heads to discriminator

        self.generator = TextGenerator(
            sequence_length, embedding_dim, d_model, num_heads
        ).build_model()

        self.discriminator = TextDiscriminator(
            sequence_length, embedding_dim, d_model, num_heads # Pass d_model and num_heads
        ).build_model()

        self.gan = self._build_gan()

    def _build_gan(self):
        # Compile the discriminator first
        self.discriminator.compile(
            optimizer=optimizers.Adam(learning_rate=5e-5), # Lowered learning rate
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.discriminator.trainable = False

        gan_input = layers.Input(shape=(self.sequence_length, self.embedding_dim))
        generated = self.generator(gan_input)
        validity = self.discriminator(generated)

        combined = models.Model(gan_input, validity, name="SentenceGAN")
        combined.compile(
            optimizer=optimizers.Adam(learning_rate=5e-5), # Lowered learning rate
            loss='binary_crossentropy' # Generator tries to fool discriminator into outputting '1'
        )

        return combined

    def train_step(self, real_data, batch_size):
        # --- Train Discriminator ---
        noise = np.random.normal(0, 1, (batch_size, self.sequence_length, self.embedding_dim))
        fake_data = self.generator.predict(noise, verbose=0)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        X = np.concatenate([real_data, fake_data])
        y = np.concatenate([real_labels, fake_labels])

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        self.discriminator.trainable = True
        d_loss = self.discriminator.train_on_batch(X, y)

        self.discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, self.sequence_length, self.embedding_dim))
        g_loss = self.gan.train_on_batch(noise, real_labels)

        return {'d_loss': d_loss[0], 'd_acc': d_loss[1], 'g_loss': g_loss}

    def train(self, data, epochs, batch_size=32):
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_batch = data[idx]

            metrics = self.train_step(real_batch, batch_size)

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: D Loss: {metrics['d_loss']:.4f}, "
                            f"D Acc: {metrics['d_acc']:.4f}, G Loss: {metrics['g_loss']:.4f}")
        logger.info("Training completed!")


def user_input(user_input_text, preprocessor_obj, tokenizer_obj, embedding_model_obj, gan_model_obj):

    logger.info(f"Processing user input: '{user_input_text}'")

    cleaned_input = preprocessor_obj.clean_text([user_input_text])
    encoded_input = preprocessor_obj.encode_texts(cleaned_input) 
    logger.info(f"Encoded input tokens (first 10): {encoded_input[0][:10]}")

    embedding_input = embedding_model_obj.predict(encoded_input, verbose=0) 
    logger.info(f"Embedding input shape: {embedding_input.shape}")

    generated_embeddings = gan_model_obj.generator.predict(embedding_input, verbose=0)
    logger.info(f"Generated embeddings shape: {generated_embeddings.shape}")

    if not isinstance(embedding_model_obj.layers[1], layers.Embedding):
        logger.error("Error: Expected Embedding layer at index 1 of embedding_model_obj for decoding.")
        print("Could not decode generated output due to model structure mismatch.")
        return

    embedding_matrix = embedding_model_obj.layers[1].get_weights()[0] 
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")

    decoded_tokens = []

    for i, embedding_vector in enumerate(generated_embeddings[0]):

        similarities = np.dot(embedding_vector, embedding_matrix.T)

        closest_token_id = np.argmax(similarities)
        decoded_tokens.append(closest_token_id)

    logger.info(f"Decoded token IDs (first 10): {decoded_tokens[:10]}")
    if tokenizer_obj.vocab_size > 3: # Ensure the ID exists in the vocab
        logger.info(f"Tokenizer decode for ID 3 (often <unk>): '{tokenizer_obj.decode([3])}'")

    # Filter out padding tokens (token ID 0) for cleaner output
    filtered_decoded_tokens = [token_id for token_id in decoded_tokens if token_id != 0]

    # Decode the sequence of token IDs back to a human-readable string
    generated_text = tokenizer_obj.decode(filtered_decoded_tokens)

    print('\n--- Results ---')
    print("Original Text:", user_input_text)
    print('Generated Output:', generated_text)


def main():

    try:
        logger.info("Loading data from sampledata.json...")
        with open('sampledata.json', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from sampledata.json. (Expected ~30,000 lines)")
    except FileNotFoundError:
        logger.error("Error: sampledata.json not found. Please ensure the file is in the same directory.")
        return 

    input_texts = [item["input"] for item in data]
    target_texts = [item["targets"] for item in data]

    preprocessor = TextPreprocessor()

    input_texts_cleaned = preprocessor.clean_text(input_texts)
    target_texts_cleaned = preprocessor.clean_text(target_texts)

    corpus = input_texts_cleaned + target_texts_cleaned
    tokenizer = preprocessor.build_tokenizer(corpus)

    token_ids = preprocessor.encode_texts(input_texts_cleaned)

    vocab_size = tokenizer.vocab_size
    embedding_dim = 256 # Increased embedding dimension
    max_len = preprocessor.max_length
    d_model = 128 # Increased d_model for attention layers
    num_heads = 8 # Consistent num_heads for attention layers

    embedding_input_layer = layers.Input(shape=(max_len,), name='embedding_input_layer')
    embedding_layer = layers.Embedding(vocab_size, embedding_dim, name='shared_embedding')(embedding_input_layer)
    embedding_model = models.Model(embedding_input_layer, embedding_layer, name='EmbeddingModel')

    embedding_features = embedding_model.predict(token_ids, verbose=0)
    logger.info(f"Embedding features shape (for GAN training): {embedding_features.shape}")

    train_data, test_data = train_test_split(
        embedding_features, test_size=0.2, random_state=42
    )

    sequence_length, current_embedding_dim = embedding_features.shape[1:]

    gan = SentenceGAN(sequence_length, current_embedding_dim, d_model=d_model, num_heads=num_heads)

    logger.info("Generator Summary:")
    gan.generator.summary()
    logger.info("\nDiscriminator Summary:")
    gan.discriminator.summary()
    logger.info("\nGAN Summary:")
    gan.gan.summary()

    gan.train(train_data, epochs=500, batch_size=32)
    logger.info("GAN Training completed!")

    user_input("The tiny seed grew slowly", preprocessor, tokenizer, embedding_model, gan)

if __name__ == "__main__":
    main()
