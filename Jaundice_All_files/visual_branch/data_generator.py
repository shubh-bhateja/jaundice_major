import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2 # In case images haven't been flattened to .npy yet and need on-the-fly loading

class MultimodalDataGenerator(Sequence):
    """
    Custom Keras Data Generator for the Multimodal Neonatal Jaundice Model.
    Inherits from tf.keras.utils.Sequence to ensure multithreading safety.
    """
    
    def __init__(self, dataset, batch_size=32, num_classes=3, visual_seq_length=50000, shuffle=True):
        """
        Initialization
        :param dataset: List of dictionaries [{'image_path', 'clinical_vector', 'tsb_value', 'risk_class'}, ...]
        :param batch_size: Number of samples per batch
        :param num_classes: Number of risk classes for one-hot encoding
        :param visual_seq_length: The expected length of the 1D CNN input
        :param shuffle: Whether to shuffle the data at the end of each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.visual_seq_length = visual_seq_length
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        
        # Initial shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data.
        """
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_visual_sequence(self, image_path):
        """
        Helper function to load the visual sequence from disk.
        Handles both raw numpy arrays (.npy) and standard images.
        """
        if image_path.endswith('.npy'):
            seq = np.load(image_path)
            
            # Ensure the sequence matches the expected length for the model
            if len(seq) > self.visual_seq_length:
                seq = seq[:self.visual_seq_length]
            elif len(seq) < self.visual_seq_length:
                # Pad with zeros if it's too short
                padding = self.visual_seq_length - len(seq)
                seq = np.pad(seq, (0, padding), 'constant', constant_values=0)
                
            # Expand dims to match (sequence_length, channels)
            seq = np.expand_dims(seq, axis=-1)
            return seq
            
        else:
            # If standard images (.jpg/.png) are provided rather than precomputed sequences
            # Note: This implies running `convert_to_1d_sequence` on-the-fly, which is slow.
            # Best practice is to save the sequences beforehand as .npy files. 
            image_bgr = cv2.imread(image_path)
            # Flatten image: (H * W * 3)
            seq = image_bgr.flatten()
            
            # Truncate or pad
            if len(seq) > self.visual_seq_length:
                 seq = seq[:self.visual_seq_length]
            elif len(seq) < self.visual_seq_length:
                 padding = self.visual_seq_length - len(seq)
                 seq = np.pad(seq, (0, padding), 'constant', constant_values=0)
                 
            seq = np.expand_dims(seq, axis=-1)
            return seq

    def __data_generation(self, batch_indices):
        """
        Generates data containing batch_size samples.
        """
        # Batch arrays
        batch_visual = []
        batch_clinical = []
        batch_tsb = []
        batch_risk = []

        for i in batch_indices:
            sample = self.dataset[i]
            
            # --- Load Inputs ---
            visual_seq = self._load_visual_sequence(sample['image_path'])
            batch_visual.append(visual_seq)
            
            clinical_vec = sample['clinical_vector']
            batch_clinical.append(clinical_vec)
            
            # --- Load Outputs ---
            batch_tsb.append(sample['tsb_value'])
            batch_risk.append(sample['risk_class'])

        # Convert lists to numpy arrays
        batch_visual = np.array(batch_visual, dtype=np.float32)
        batch_clinical = np.array(batch_clinical, dtype=np.float32)
        batch_tsb = np.array(batch_tsb, dtype=np.float32)
        
        # Convert risk classes to one-hot encoding
        # E.g. Risk class '2' becomes [0, 0, 1]
        batch_risk = np.array(batch_risk, dtype=np.int32)
        batch_risk_one_hot = tf.keras.utils.to_categorical(batch_risk, num_classes=self.num_classes)

        # Build dictionary map to matching Keras naming convention from `model_integration.py`
        X = {
            'visual_input': batch_visual,
            'clinical_input': batch_clinical
        }

        y = {
            'tsb_regression_output': batch_tsb,
            'risk_classification_output': batch_risk_one_hot
        }

        return X, y

if __name__ == "__main__":
    # Quick Test Execution
    # Create fake dataset generated by `data_loader.py`
    fake_dataset = [
        {
            'image_path': 'fake_path.npy', 
            'clinical_vector': np.array([2.5, 38, 48]), 
            'tsb_value': 12.5, 
            'risk_class': 1
        } for _ in range(10) # 10 samples
    ]
    
    # We must patch load functionality to test without real files
    def dummy_load(path): return np.zeros((1000, 1))
    
    generator = MultimodalDataGenerator(
        dataset=fake_dataset,
        batch_size=4,
        num_classes=3,
        visual_seq_length=1000
    )
    generator._load_visual_sequence = dummy_load
    
    # Fetch one batch
    X_batch, y_batch = generator[0]
    
    print("Generated Inputs:")
    print("  Visual Input Shape:   ", X_batch['visual_input'].shape)
    print("  Clinical Input Shape: ", X_batch['clinical_input'].shape)
    
    print("\nGenerated Outputs:")
    print("  TSB Regression Shape:       ", y_batch['tsb_regression_output'].shape)
    print("  Risk Classification Shape:  ", y_batch['risk_classification_output'].shape)
