import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input

def build_visual_branch_1dcnn(sequence_length=None, channels=1):
    """
    Builds the 1DCNN model for the Visual Branch as described in Table 2 of the research paper.
    
    Parameters:
    - sequence_length: The length of the 1D sequence from the preprocessing mask.
                       Set to a fixed integer if sequences are padded/truncated to a uniform length.
                       Set to None if the model should accept variable length sequences.
    - channels: The number of features per step. Since the preprocessing flattened the array,
                we treat each element as a single step with 1 channel.
                
    Returns:
    - model: A compiled tf.keras.Model
    """
    
    model = Sequential(name="Visual_Branch_1DCNN")
    
    # Input Layer
    # Based on preprocess.py which produces a flattened 1D Array of size N * 3
    # The input shape is (sequence_length, 1)
    model.add(Input(shape=(sequence_length, channels), name="visual_input"))
    
    # 1. First Conv1D block (32 filters)
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', name="conv1d_1"))
    # 2. MaxPooling1D after each convolution
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_1"))
    
    # 3. Second Conv1D block (64 filters)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name="conv1d_2"))
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_2"))
    
    # 4. Third Conv1D block (128 filters)
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name="conv1d_3"))
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_3"))
    
    # 5. Fourth Conv1D block (128 filters)
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name="conv1d_4"))
    model.add(MaxPooling1D(pool_size=2, name="maxpool1d_4"))
    
    # 6. Add a Flatten layer at the end to generate the Visual Embedding Vector
    model.add(Flatten(name="visual_embedding"))
    
    return model

if __name__ == "__main__":
    # Example usage:
    # If the output from preprocess.py has a fixed padded/truncated length of e.g. 50,000 values:
    # This value heavily depends on the average number of skin pixels detected in your dataset.
    # We use 50_000 as a placeholder value below.
    EXPECTED_SEQUENCE_LENGTH = 50000 
    
    model = build_visual_branch_1dcnn(sequence_length=EXPECTED_SEQUENCE_LENGTH)
    
    print("Visual Branch 1DCNN Architecture:")
    model.summary()
