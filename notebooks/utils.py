import librosa
import numpy as np
from typing import Tuple, Callable




def compute_intra_song_similarity(audio_file: str, similarity_func: Callable) -> float:
    """
    Compute similarity between the first two-thirds and the last two-thirds of an audio file.
    
    Parameters:
    - audio_file (str): Path to the audio file.
    - sr (int): Sampling rate to use when loading the audio file.
    - similarity_func (function): A function that computes similarity between two audio segments.
    
    Returns:
    - float: The computed similarity value.
    """
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Split the audio into two segments: first two-thirds and last two-thirds
    two_thirds_length = int(2 * len(y) / 3)
    segment1 = y[:two_thirds_length]
    segment2 = y[len(y) - two_thirds_length:]
    
    # Compute the similarity between the two segments
    similarity = similarity_func(segment1, sr, segment2, sr)
    
    return similarity


def align_sequences(seq1, seq2) -> Tuple[np.ndarray, np.ndarray]:
    """Align two sequences by padding the shorter sequence."""
    diff = len(seq1) - len(seq2)
    if diff > 0:
        seq2 = np.pad(seq2, (0, diff), 'constant')
    else:
        seq1 = np.pad(seq1, (0, -diff), 'constant')
    return seq1, seq2


def resample_to_match(y1: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resample both signals to have the same length based on the shorter signal."""
    
    target_length = min(len(y1), len(y2))
    y1_resampled = y1[:target_length]
    y2_resampled = y2[:target_length]
    
    return y1_resampled, y2_resampled