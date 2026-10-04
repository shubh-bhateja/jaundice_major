import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate

def build_multimodal_model(visual_seq_length, clinical_features_count):
    """
    Builds the multimodal neural network using the Keras Functional API.
    Combines the Visual Branch (1DCNN) and Clinical Branch (Dense).
    Produces two outputs: TSB Regression and Risk Classification.
    """
    
    # ==========================================
    # 1. Visual Branch
    # ==========================================
    visual_input = Input(shape=(visual_seq_length, 1), name="visual_input")
    
    # Block 1: 32 Filters
    x_vis = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name="v_conv1d_1")(visual_input)
    x_vis = MaxPooling1D(pool_size=2, name="v_maxpool_1")(x_vis)
    
    # Block 2: 64 Filters
    x_vis = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="v_conv1d_2")(x_vis)
    x_vis = MaxPooling1D(pool_size=2, name="v_maxpool_2")(x_vis)
    
    # Block 3: 128 Filters
    x_vis = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name="v_conv1d_3")(x_vis)
    x_vis = MaxPooling1D(pool_size=2, name="v_maxpool_3")(x_vis)
    
    # GlobalAveragePooling1D replaces Flatten for better generalized feature representation
    visual_embedding = GlobalAveragePooling1D(name="visual_embedding")(x_vis)
    
    
    # ==========================================
    # 2. Clinical Branch
    # ==========================================
    clinical_input = Input(shape=(clinical_features_count,), name="clinical_input")
    
    # Dense Block 1: 64 Units
    x_clin = Dense(64, activation='relu', name="c_dense_1")(clinical_input)
    x_clin = Dropout(0.4, name="c_dropout_1")(x_clin)
    
    # Dense Block 2: 32 Units
    clinical_embedding = Dense(32, activation='relu', name="clinical_embedding")(x_clin)
    
    
    # ==========================================
    # 3. Embedding Fusion
    # ==========================================
    # Concatenate the visual and clinical embeddings
    fused_features = Concatenate(name="embedding_fusion")([visual_embedding, clinical_embedding])
    
    
    # ==========================================
    # 4. Fully Connected Layers & Output Heads
    # ==========================================
    # Shared Fully Connected Layers post-fusion
    x_fused = Dense(128, activation='relu', name="fc_shared_1")(fused_features)
    x_fused = Dropout(0.3, name="fc_dropout_1")(x_fused)
    
    x_fused = Dense(64, activation='relu', name="fc_shared_2")(x_fused)
    
    # Output Head 1: TSB Regression (Continuous output -> linear activation)
    tsb_output = Dense(1, activation='linear', name="tsb_regression_output")(x_fused)
    
    # Output Head 2: Risk Classification (Probabilities -> softmax activation or sigmoid depending on classes)
    # Assuming standard Risk Classifications (e.g. Low, Medium, High Risk), we use softmax with 3 units.
    # Change units=3 to the number of classes you have. If it's binary, use units=1 and activation='sigmoid'.
    risk_classes_count = 3 
    risk_output = Dense(risk_classes_count, activation='softmax', name="risk_classification_output")(x_fused)
    
    
    # ==========================================
    # Compile Model
    # ==========================================
    # Create the functional model defining the inputs and dual outputs
    model = Model(
        inputs=[visual_input, clinical_input], 
        outputs=[tsb_output, risk_output],
        name="Multimodal_Neonatal_Jaundice_Model"
    )
    
    # Compile with MSE for regression and Categorical Cross-Entropy for classification
    # Weights can be adjusted to balance the importance of each loss during training
    model.compile(
        optimizer='adam',
        loss={
            'tsb_regression_output': 'mse',
            'risk_classification_output': 'categorical_crossentropy'  # Use 'sparse_categorical_crossentropy' if labels are integers
        },
        loss_weights={
            'tsb_regression_output': 1.0,
            'risk_classification_output': 1.0
        },
        metrics={
            'tsb_regression_output': ['mae'], # Mean Absolute Error for tracking
            'risk_classification_output': ['accuracy']
        }
    )
    
    return model

if __name__ == "__main__":
    # Example usage / Dimension checks
    VISUAL_SEQ_LEN = 10000     # Replace with your actual flattened 1D array length
    CLINICAL_FEATURES = 10     # Replace with your actual number of clinical parameters (e.g. GA, Weight, Age in hrs)
    
    multimodal_model = build_multimodal_model(
        visual_seq_length=VISUAL_SEQ_LEN,
        clinical_features_count=CLINICAL_FEATURES
    )
    
    # Print the model architecture
    multimodal_model.summary()
