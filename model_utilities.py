# Import necessary libraries
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Attention, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ======== DATA LOADING AND PREPROCESSING ========
def load_and_preprocess_data(train_data, val_data, max_features=10000):
    """
    Load and preprocess the toxic comment dataset.
    
    Args:
        train_data (str): DataFrame to the CSV file containing the training data
        val_data (str): DataFrame to the CSV file containing the validation data
        max_features (int): Maximum number of words to keep in the vocabulary
        
    Returns:
        dict: Dictionary containing processed data and metadata
    """
    print("Loading and preprocessing data...")
    
    # Load the data
    train_data = train_data.drop('comment_text', axis=1)

    # Load the validation data
    val_data = val_data.drop('comment_text', axis=1)
    
    # Extract features and labels
    X_train = train_data['cleaned_comment']
    y_train = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Extract features and labels
    X_val = val_data['cleaned_comment']
    y_val = val_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    
    # Tokenize the train text data
    tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    
    # Get sequence length
    seq_lengths = [len(x) for x in sequences]

    # Choose max_seq_length based on the 95th percentile
    max_seq_length = int(np.percentile(seq_lengths, 95))
    
    # Pad the train sequences
    X_train_padded = pad_sequences(sequences, maxlen=max_seq_length)

    # Tokenize the val text data
    sequences = tokenizer.texts_to_sequences(X_val)

    # Pad the val sequences
    X_val_padded = pad_sequences(sequences, maxlen=max_seq_length)

    # Create result dictionary
    result = {
        'X_train': X_train_padded,
        'X_val': X_val_padded,
        'y_train': y_train,
        'y_val': y_val,
        'tokenizer': tokenizer,
        'max_seq_length': max_seq_length,
        'word_index': tokenizer.word_index,
        'max_features': max_features
    }
    
    print(f"Train set shape: {X_train_padded.shape}")
    print(f"Test set shape: {X_val_padded.shape}")
    
    return result

# ======== GLOVE EMBEDDINGS WITH 6B PARAMETERS ========
def load_glove6B_embeddings(word_index, embedding_dim=100, max_features=10000):
    """
    Load GloVe embeddings and create an embedding matrix.
    
    Args:
        word_index (dict): Word index dictionary from the tokenizer
        embedding_dim (int): Dimension of the embeddings to use
        max_features (int): Maximum number of words to keep
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    print(f"\nLoading GloVe embeddings (dimension: {embedding_dim})...")
    
    # Prepare the embedding matrix
    vocab_size = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Path to the GloVe embeddings file
    glove_path = f'/kaggle/input/glove-embeddings/glove.6B.{embedding_dim}d.txt'
    
    # Read the GloVe embeddings
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    print(f"Found {len(embeddings_index):,} word vectors in GloVe.")
    
    # Create the embedding matrix
    num_words_found = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            num_words_found += 1
        else:
            # No match - leave as random init
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    print(f"Found embeddings for {num_words_found:,} / {min(vocab_size, len(word_index)):,} words.")
    
    return embedding_matrix

# ======== LOAD GLOVE EMBEDDINGS WITH 840B PARAMETERS ========
def load_glove840B_embeddings(word_index, embedding_dim=300, max_features=100000):
    """
    Load pre-pickled GloVe embeddings and create an embedding matrix.
    
    Args:
        word_index (dict): Word index dictionary from the tokenizer
        embedding_dim (int): Dimension of the embeddings (must match pickle file)
        max_features (int): Maximum number of words to keep
        
    Returns:
        numpy.ndarray: Embedding matrix
    """
    print(f"\nLoading pickled GloVe embeddings (dimension: {embedding_dim})...")
    
    # Prepare the embedding matrix
    vocab_size = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Path to the pickled GloVe embeddings file
    glove_path = '/kaggle/input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
    
    try:
        # Load the pickled embeddings
        with open(glove_path, 'rb') as f:
            embeddings_index = pickle.load(f)
            
        print(f"Successfully loaded {len(embeddings_index):,} word vectors from pickle file.")
        
        # Verify embedding dimension matches
        sample_vector = next(iter(embeddings_index.values()))
        if len(sample_vector) != embedding_dim:
            print(f"Warning: Expected dimension {embedding_dim} but got {len(sample_vector)}")
            embedding_dim = len(sample_vector)
            embedding_matrix = np.zeros((vocab_size, embedding_dim))
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise
    
    # Create the embedding matrix
    num_words_found = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
            
        # Handle different word cases and variations
        embedding_vector = None
        for word_variant in [word, word.lower(), word.capitalize(), word.upper()]:
            if word_variant in embeddings_index:
                embedding_vector = embeddings_index[word_variant]
                break
                
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            num_words_found += 1
        else:
            # Use random initialization with smaller scale
            embedding_matrix[i] = np.random.normal(scale=0.1, size=(embedding_dim,))
    
    print(f"Found embeddings for {num_words_found:,} / {min(vocab_size, len(word_index)):,} words.")
    
    return embedding_matrix

# ======== FOCAL LOSS FUNCTION ========
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss implementation for handling class imbalance.
    Focuses learning on hard examples and down-weights easy examples.
    
    Args:
        alpha: Weighting factor for rare class (typically 0.25)
        gamma: Focusing parameter (typically 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal loss
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        
        return tf.reduce_mean(focal_loss)
    
# ======== ATTENTION MECHANISM ========
class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention mechanism layer for LSTM.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # x: [batch_size, seq_len, features]
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()
    
# ======== MODEL BUILDING ========
def build_lstm_model(embedding_matrix, max_seq_length, vocab_size, use_focal_loss=False, alpha=0.25, learning_rate=0.003, embedding_dim=100):
    """
    Build LSTM model with embedding matrix and class imbalance handling.
    
    Args:
        embedding_matrix: Pre-trained embedding matrix
        max_seq_length: Maximum sequence length
        vocab_size: Size of vocabulary
        use_focal_loss: Whether to use focal loss
        alpha: Weighting factor for rare class
        learning_rate: Learning rate for the optimizer
        embedding_dim: Embedding dimension
    
    Returns:
        Compiled model with appropriate loss function
    """
    print("\nBuilding model for the dataset...")
    
    # Input layer
    inputs = Input(shape=(max_seq_length,))
    
    # Embedding layer
    x = Embedding(
        input_dim=vocab_size,   
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_length,
        trainable=False
    )(inputs)
    
    # Bidirectional LSTM layers
    lstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Attention
    att = AttentionLayer()(lstm)
    x = Dropout(0.2)(att)
    
    # Dense layers with batch normalization
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer 
    outputs = Dense(6, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Choose loss function based on strategy
    if use_focal_loss:
        print("Using Focal Loss for class imbalance")
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        print("Using standard Binary Crossentropy")
        loss_fn = 'binary_crossentropy'
    
    # Compile with appropriate loss
    model.compile(
        loss=loss_fn,
        optimizer= Adam(learning_rate=learning_rate),
        metrics=['accuracy', AUC(name='roc_auc')]
    )
    
    print(model.summary())
    return model

# ======== MODEL BUILDING WITH SELF-ATTENTION ========
def build_lstm_model_self_attention(embedding_matrix, max_seq_length, vocab_size, use_focal_loss=False, embedding_dim=100):
    """
    Build Bi-LSTM model with TensorFlow's built-in Attention layer and GlobalAveragePooling1D layer with handling class imbalance handling.
    
    Args:
        embedding_matrix: Pre-trained embedding matrix
        max_seq_length: Maximum sequence length
        vocab_size: Size of vocabulary
        use_focal_loss: Whether to use focal loss
        embedding_dim: Embedding dimension
    
    Returns:
        Compiled model with appropriate loss function
    """
    print("\nBuilding model using TensorFlow ...")
    
    # Input layer
    inputs = Input(shape=(max_seq_length,))
    
    # Embedding layer - converts word indices to dense vectors
    x = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_length,
        trainable=False  # Keep pre-trained embeddings frozen
    )(inputs)
    
    # Bidirectional LSTM layers - processes sequences in both directions
    lstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Self-Attention
    # This allows the model to focus on different parts of the sequence
    attention_output = Attention()([lstm, lstm])  # Self-attention: query=key=value=lstm
    
    # Global average pooling to reduce sequence dimension
    # This replaces custom weighted sum operation
    x = GlobalAveragePooling1D()(attention_output)
    x = Dropout(0.2)(x)
    
    # Dense layers with batch normalization for final classification
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer for multi-label classification (6 classes)
    outputs = Dense(6, activation='sigmoid')(x)
    
    # Create the complete model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Choose loss function based on class imbalance strategy
    if use_focal_loss:
        print("Using Focal Loss for class imbalance")
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        print("Using standard Binary Crossentropy")
        loss_fn = 'binary_crossentropy'
    
    # Compile with appropriate loss and metrics
    model.compile(
        loss=loss_fn,
        optimizer=Adam(learning_rate=0.03),
        metrics=['accuracy', AUC(name='roc_auc')]
    )
    
    print(model.summary())
    return model

# ======== MODEL BUILDING WITH MULTI-HEAD ATTENTION ========
def build_lstm_model_with_multi_head_attention(embedding_matrix, max_seq_length, vocab_size, use_focal_loss=False, embedding_dim=100):
    """
    Alternative implementation using MultiHeadAttention for more sophisticated attention.
    This gives you more control and is closer to modern transformer architectures.
    """
    print("\nBuilding model with MultiHeadAttention layer...")
    
    inputs = Input(shape=(max_seq_length,))
    
    # Embedding layer
    x = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_seq_length,
        trainable=False
    )(inputs)
    
    # Bidirectional LSTM
    lstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    
    # Multi-head self-attention - more powerful than single-head attention
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=8,          # Number of attention heads
        key_dim=64,           # Dimension of each attention head
        dropout=0.1
    )
    attention_output = attention_layer(lstm, lstm)  # Self-attention
    
    # Add residual connection and layer normalization (like in Transformers)
    attention_output = tf.keras.layers.Add()([lstm, attention_output])
    attention_output = tf.keras.layers.LayerNormalization()(attention_output)
    
    # Global average pooling to get fixed-size representation
    x = GlobalAveragePooling1D()(attention_output)
    x = Dropout(0.2)(x)
    
    # Classification layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(6, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Loss function selection
    if use_focal_loss:
        print("Using Focal Loss for class imbalance")
        loss_fn = FocalLoss(alpha=0.70, gamma=2.0)
    else:
        print("Using standard Binary Crossentropy")
        loss_fn = 'binary_crossentropy'
    
    model.compile(
        loss=loss_fn,
        optimizer=Adam(learning_rate=0.03),
        metrics=['accuracy', AUC(name='roc_auc')]
    )
    
    print(model.summary())
    return model

# ======== TRAINING FUNCTION ========
def train_model(model, data_dict, batch_size=32, epochs=10):
    """    
    Args:
        model: Compiled model
        data_dict: Dictionary with training and validation data
        batch_size: Batch size
        epochs: Number of epochs
    
    Returns:
        Trained model and history
    """
    print("\nTraining model with class imbalance handling...")
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # Callbacks durning the training process
    callbacks = [
        EarlyStopping(
            monitor='val_roc_auc',  # Monitor AUC
            patience=5,             # 5 epochs to stop if the auc is not improved   
            mode='max',             # Maximize AUC
            restore_best_weights=True, # Restore the best weights
            verbose=1                # Show the progress
        ),
        ReduceLROnPlateau(
            monitor='val_roc_auc', # Monitor AUC
            factor=0.3,            # Reduce the learning rate by 30%
            patience=3,            # 3 epochs to reduce the learning rate by 30% if the auc is not improved
            mode='max',            # Maximize AUC
            min_lr=1e-6,           # The learning rate will not go below 1e-6
            verbose=1              
        ),
        ModelCheckpoint(
            'best_model.keras',    # save model with the name 
            monitor='val_roc_auc', # save model based on the AUC score monitored
            save_best_only=True,   
            mode='max',            # save model with the max AUC score only
            verbose=1
        )
    ]
    
    # Train with smaller batch size for better gradient estimates
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,  # Smaller batch size for imbalanced data
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ======== FIND OPTIMAL THRESHOLD ========
def find_optimal_threshold(y_true, y_pred):
    """Find threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    return thresholds[np.nanargmax(f1_scores)]

# ======== MODEL EVALUATION ========
def evaluate_model(model, data_dict, threshold='optimal'):
    """
    Evaluate model with optimal threshold per label
    threshold: 'optimal' or float (0-1) for fixed threshold
    """
    print("\nEvaluating the model...")
    
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    y_pred = model.predict(X_val)
    
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    metrics = {
        'category_auc': {},
        'optimal_thresholds': {},
        'classification_reports': {},
        'overall_auc': 0.0
    }

    # Calculate optimal thresholds if requested
    if threshold == 'optimal':
        optimal_thresholds = [find_optimal_threshold(y_val[:, i], y_pred[:, i]) 
                             for i in range(len(categories))]
    else:
        optimal_thresholds = [threshold] * len(categories)

    # Convert probabilities to binary predictions
    y_pred_binary = np.zeros_like(y_pred)
    for i in range(len(categories)):
        y_pred_binary[:, i] = (y_pred[:, i] >= optimal_thresholds[i]).astype(int)

    # Generate metrics for each category
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        # Store thresholds
        metrics['optimal_thresholds'][category] = optimal_thresholds[i]
        
        # Calculate AUC
        auc = roc_auc_score(y_val[:, i], y_pred[:, i])
        metrics['category_auc'][category] = auc
        
        # Generate classification report
        report = classification_report(
            y_val[:, i], y_pred_binary[:, i],
            target_names=[f'Non-{category}', category],
            output_dict=True
        )
        metrics['classification_reports'][category] = report
        
        # Print results
        print(f"\n=== {category.upper()} ===")
        print(f"Optimal Threshold: {optimal_thresholds[i]:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(classification_report(
            y_val[:, i], y_pred_binary[:, i],
            target_names=[f'Non-{category}', category]
        ))
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_val[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{category} (AUC={auc:.2f})')

    # Finalize plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Category')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves.png')
    plt.show()

    # Calculate overall metrics
    metrics['overall_auc'] = np.mean(list(metrics['category_auc'].values()))
    print(f"\nOverall ROC AUC: {metrics['overall_auc']:.4f}")
    
    return metrics

# ======== VISUALIZATION ========
def plot_training_history(history):
    """
    Plot the training and validation AUC and loss and accuracy.
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 5))

    # Plot roc_auc
    plt.subplot(1, 2, 1)
    plt.plot(history.history['roc_auc'], label='Training AUC')
    plt.plot(history.history['val_roc_auc'], label='Validation AUC')
    plt.title('AUC over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # # Plot accuracy
    # plt.subplot(1, 3, 3)
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ======== SAVE AND LOAD FUNCTIONS ========
def save_model_and_data(model, data_dict, model_path='toxic_model.keras', data_path='preprocessing_data.pkl'):
    """
    Save the model and preprocessing data.
    
    Args: 
        model (tensorflow.keras.models.Model): The trained model
        data_dict (dict): Dictionary with preprocessing data
        model_path (str): Path to save the model
        data_path (str): Path to save preprocessing data
    """
    print(f"\nSaving model to {model_path}")
    model.save(model_path)
    
    # Extract only the necessary preprocessing data
    preprocessing_data = {
        'tokenizer': data_dict['tokenizer'],
        'max_seq_length': data_dict['max_seq_length'],
        'word_index': data_dict['word_index'],
        'max_features': data_dict['max_features']
    }
    
    print(f"Saving preprocessing data to {data_path}")
    with open(data_path, 'wb') as f:
        pickle.dump(preprocessing_data, f)
        
# ======== PREDICTION FUNCTION ========
def predict_toxicity(text, model, data_dict):
    """
    Predict toxicity scores for a new comment.
    
    Args:
        text (str): The comment text to classify
        model (tensorflow.keras.models.Model): The trained model
        data_dict (dict): Dictionary containing processing data
        
    Returns:
        dict: Toxicity scores for each category
    """
    # Preprocess the text
    tokenizer = data_dict['tokenizer']
    max_seq_length = data_dict['max_seq_length']
    
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_seq_length)
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0]
    
    # Create result dictionary
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = {category: float(score) for category, score in zip(categories, prediction)}
    
    return result

# ======== LOAD MODEL AND DATA ========
def load_model_and_data(model_path='toxic_model.keras', data_path='preprocessing_data.pkl'):
    """
    Load the saved model and preprocessing data.
    
    Args:
        model_path (str): Path to the saved model
        data_path (str): Path to the saved preprocessing data
        
    Returns:
        tuple: Loaded model and preprocessing data dictionary
    """
    print(f"\nLoading model from {model_path}")
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    
    print(f"Loading preprocessing data from {data_path}")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    return model, data_dict

# ======== LOAD AND PREPROCESS TEST DATA ========
def load_and_preprocess_test_data(df_test, tokenizer, max_seq_length):
    """
    Load and preprocess the test data.
    
    Args:   
        tokenizer (Tokenizer): Tokenizer object
        max_seq_length (int): Maximum sequence length
    """
    print("Loading and preprocessing test data...")
    
    # Load the data
    data = df_test.drop('comment_text', axis=1)
    # data['cleaned_comment'] = data['cleaned_comment'].fillna('you are so kind, thank you for helping me')
    
    # Extract features and labels
    X = data['cleaned_comment']
    
    # Tokenize the text data
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=max_seq_length)

    return X_padded, data



