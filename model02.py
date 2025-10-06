import numpy as np
import re
import json
import logging
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, vocab_size=68000, max_seq_length=512):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_built = False
    
    def cleantext(self, texts):
        cleandata = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text) 
            text = re.sub(r'\s+', ' ', text).strip() 
            cleandata.append(text)
        return cleandata
    
    def build_vocabulary(self, texts):
        """Build vocabulary from text data"""
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Create vocabulary with most common words
        most_common = word_counts.most_common(self.vocab_size - 4)  # Reserve 4 special tokens
        
        # Add special tokens
        self.word_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        
        # Add regular vocabulary
        for i, (word, _) in enumerate(most_common, start=4):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
        
        self.vocab_built = True
        logger.info(f"Built vocabulary with {len(self.word_to_id)} tokens")
    
    def text_to_sequence(self, text):
        """Convert text to sequence of token IDs"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = text.split()
        sequence = [self.word_to_id.get('<START>')]
        for word in words:
            sequence.append(self.word_to_id.get(word, self.word_to_id['<UNK>']))
        sequence.append(self.word_to_id.get('<END>'))
        
        # Pad or truncate to max_seq_length
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence.extend([self.word_to_id['<PAD>']] * (self.max_seq_length - len(sequence)))
        
        return sequence
    
    def sequences_to_text(self, sequences):
        """Convert sequences back to text"""
        texts = []
        for seq in sequences:
            words = []
            for token_id in seq:
                word = self.id_to_word.get(token_id, '<UNK>')
                if word == '<END>':
                    break
                if word not in ['<PAD>', '<START>']:
                    words.append(word)
            texts.append(' '.join(words))
        return texts

class MultiHeadAttentionWithRoPE(layers.Layer):
    def __init__(self, d_model, num_heads, name="MultiHeadAttentionWithRoPE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model, name='query_projection')
        self.wk = layers.Dense(d_model, name='key_projection')
        self.wv = layers.Dense(d_model, name='value_projection')
        self.dense = layers.Dense(d_model, name='output_projection')

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def rotate_half(self, x):
        x1, x2 = tf.split(x, 2, axis=-1)
        return tf.concat([-x2, x1], axis=-1)

    def apply_rope(self, q, k, seq_len):
        pos = tf.cast(tf.range(seq_len), dtype=tf.float32)
        inv_freq = 1.0 / (10000**(tf.cast(tf.range(0, self.depth, 2), dtype=tf.float32) / self.depth))
        angle = tf.einsum('i,j->ij', pos, inv_freq)
        
        cos_part = tf.repeat(tf.cos(angle), 2, axis=-1)
        sin_part = tf.repeat(tf.sin(angle), 2, axis=-1)
        
        cos_part = cos_part[tf.newaxis, tf.newaxis, :, :]
        sin_part = sin_part[tf.newaxis, tf.newaxis, :, :]
        q_rotated = q * cos_part + self.rotate_half(q) * sin_part
        k_rotated = k * cos_part + self.rotate_half(k) * sin_part
        
        return q_rotated, k_rotated

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            if len(inputs) == 3:
                q, k, v = inputs
            elif len(inputs) == 1:
                q = k = v = inputs[0]
            else:
                raise ValueError("inputs should be a list of 1 or 3 tensors")
        else:
            q = k = v = inputs
            
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        q = self.wq(q)  
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        q, k = self.apply_rope(q, k, seq_len)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)  
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, tf.newaxis, tf.newaxis, :]
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)  
        output = tf.transpose(output, perm=[0, 2, 1, 3])  
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  
        output = self.dense(concat_attention)

        return output

class SwiGLULayer(layers.Layer):
    def __init__(self, name="swiglu", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        x1, x2 = tf.split(inputs, 2, axis=-1)
        return tf.nn.silu(x1) * x2

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def transformer_model(vocab_size, hidden_size, num_layers, num_heads, max_seq_length=512):
    inputs = keras.Input(shape=(None,), dtype=tf.int32, name='input_tokens')
    
    padding_mask = layers.Lambda(create_padding_mask, name='padding_mask')(inputs)
    
    x = layers.Embedding(vocab_size, hidden_size, name='token_embedding')(inputs)
    
    for i in range(num_layers):
        attn_output = MultiHeadAttentionWithRoPE(
            hidden_size, 
            num_heads, 
            name=f'attention_layer_{i}'
        )(x, mask=padding_mask)
        
        x = layers.LayerNormalization(name=f'norm_1_{i}')(x + attn_output)

        ffn_intermediate = layers.Dense(hidden_size * 2, name=f'ffn_intermediate_{i}')(x)
        ffn_intermediate = SwiGLULayer(name=f'swiglu_{i}')(ffn_intermediate)
        ffn_output = layers.Dense(hidden_size, name=f'ffn_output_{i}')(ffn_intermediate)
        
        x = layers.LayerNormalization(name=f'norm_2_{i}')(x + ffn_output)
    
    outputs = layers.Dense(vocab_size, name='output_projection')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name='transformer_with_rope')

def prepare_training_data(data, text_processor):
    """Prepare training data from JSON format"""
    input_sequences = []
    target_sequences = []
    
    for item in data:
        input_text = item['input']
        targets = item['targets']
        
        # Clean and process input
        input_cleaned = text_processor.cleantext([input_text])[0]
        input_seq = text_processor.text_to_sequence(input_cleaned)
        
        # Process each target
        for target in targets:
            target_cleaned = text_processor.cleantext([target])[0]
            target_seq = text_processor.text_to_sequence(target_cleaned)
            
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)
    
    return np.array(input_sequences), np.array(target_sequences)

def create_dataset(input_sequences, target_sequences, batch_size=32):
    """Create TensorFlow dataset"""
    # For language modeling, we use target sequences shifted by one
    input_data = target_sequences[:, :-1]  # All tokens except last
    target_data = target_sequences[:, 1:]   # All tokens except first
    
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    # If you have a JSON file, uncomment this:
    with open("sampledata.json", encoding='utf-8') as f:
        sample_data = json.load(f)

    # Model parameters
    VOCAB_SIZE = 10000  # Reduced for small dataset
    HIDDEN_SIZE = 512   # Reduced for faster training
    NUM_LAYERS = 6      # Reduced for faster training
    NUM_HEADS = 8       # Reduced accordingly
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 8
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    
    print(f"Creating transformer model with:")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Initialize text processor
    text_processor = TextProcessor(vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    
    # Collect all text for vocabulary building
    all_texts = []
    for item in sample_data:
        all_texts.append(item['input'])
        all_texts.extend(item['targets'])
    
    # Clean and build vocabulary
    cleaned_texts = text_processor.cleantext(all_texts)
    text_processor.build_vocabulary(cleaned_texts)
    
    # Prepare training data
    print("Preparing training data...")
    input_sequences, target_sequences = prepare_training_data(sample_data, text_processor)
    
    print(f"Training data shape: {input_sequences.shape}")
    print(f"Target data shape: {target_sequences.shape}")
    
    # Split into train/validation
    train_input, val_input, train_target, val_target = train_test_split(
        input_sequences, target_sequences, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = create_dataset(train_input, train_target, BATCH_SIZE)
    val_dataset = create_dataset(val_input, val_target, BATCH_SIZE)
    
    # Create model
    model = transformer_model(
        vocab_size=len(text_processor.word_to_id),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('final_model.h5')
    print("Model saved successfully!")
    
    # Test the model with a sample
    print("\nTesting the model:")
    test_input = "The scrum master facilitated the daily stand-up meeting."
    test_cleaned = text_processor.cleantext([test_input])[0]
    test_seq = np.array([text_processor.text_to_sequence(test_cleaned)])
    
    # Generate prediction
    prediction = model.predict(test_seq[..., :-1])
    predicted_tokens = np.argmax(prediction[0], axis=-1)
    
    # Convert back to text
    predicted_text = text_processor.sequences_to_text([predicted_tokens])[0]
    print(f"Input: {test_input}")
    print(f"Generated: {predicted_text}")
    
    return model, text_processor, history

if __name__ == '__main__':
    model, processor, history = main()