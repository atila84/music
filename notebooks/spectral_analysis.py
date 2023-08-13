import librosa
import numpy as np
from utils import compute_intra_song_similarity, align_sequences, resample_to_match


# Spectrogram
def compute_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute the magnitude spectrogram of an audio signal.
    
    Parameters:
        y (np.ndarray): The audio time series.
        sr (int): The sample rate of the audio time series.
        
    Returns:
        np.ndarray: The magnitude spectrogram.
    """
    # Compute the short-time Fourier transform (STFT) of the audio
    D = np.abs(librosa.stft(y))
    # Convert the STFT to a magnitude spectrogram
    return librosa.amplitude_to_db(D, ref=np.max)



def spectrogram_similarity(spectrogram1: np.ndarray, spectrogram2: np.ndarray) -> float:
    """Compute similarity between two spectrograms using MSE.
    
    Parameters:
        spectrogram1 (np.ndarray): The magnitude spectrogram of the first audio.
        spectrogram2 (np.ndarray): The magnitude spectrogram of the second audio.
        
    Returns:
        float: The mean squared error between the two spectrograms.
    """
    # Ensure both spectrograms are of the same shape by trimming or padding
    min_time = min(spectrogram1.shape[1], spectrogram2.shape[1])
    spectrogram1 = spectrogram1[:, :min_time]
    spectrogram2 = spectrogram2[:, :min_time]

    mse = np.mean((spectrogram1 - spectrogram2)**2)
    print("mse = ", mse)
    
    # For similarity, we'd often want a measure where higher values indicate more similarity.
    # So, we can transform the MSE into a similarity measure by taking its inverse or negative.
    similarity = 1 / (1 + np.exp(mse / -10.0))
    
    return similarity


def spectrogram_cosine_similarity(spectrogram1: np.ndarray, spectrogram2: np.ndarray) -> float:
    """Compute the cosine similarity between the spectrograms of two audio signals.
    
    Parameters:
        y1 (np.ndarray): The first audio signal.
        sr1 (int): The sample rate of the first audio signal.
        y2 (np.ndarray): The second audio signal.
        sr2 (int): The sample rate of the second audio signal.

    Returns:
        float: The cosine similarity between the two spectrograms.
    """
    # # Compute the spectrograms
    # spec1 = compute_spectrogram(y1, sr1)
    # spec2 = compute_spectrogram(y2, sr2)
    
    # Flatten the spectrogram matrices to make them vectors
    spec1_vec = spectrogram1.flatten()
    spec2_vec = spectrogram2.flatten()

    # Ensure that both vectors have the same shape
    min_len = min(len(spec1_vec), len(spec2_vec))
    spec1_vec = spec1_vec[:min_len]
    spec2_vec = spec2_vec[:min_len]
    
    # Compute cosine similarity
    similarity = cosine_similarity(spec1_vec.reshape(1, -1), spec2_vec.reshape(1, -1))

    return similarity[0][0]



def compute_spectrogram_similarity(user_y: np.ndarray, user_sr: int, 
                                   app_y: np.ndarray, app_sr: int, 
                                   use_cosine: bool=True) -> float:

    """Compute similarity between two frames.
    
    Parameters:
        :param user_y: User's audio time series.
        :param user_sr: User's audio sampling rate.
        :param app_y: Reference audio time series.
        :param app_sr: Reference audio sampling rate.
    Returns:
        float: sim score.
    """

    user_spec = compute_spectrogram(user_y, user_sr)
    app_spec = compute_spectrogram(app_y, app_sr)

    # print('user_spec = ', user_spec.shape)

    if use_cosine:
        return spectrogram_cosine_similarity(user_spec, app_spec)

    return spectrogram_similarity(user_spec, app_spec)




# Spectral Centroid: 
def extract_spectral_centroid(y: np.ndarray, sr: int) -> float:
    """
    Extracts the mean spectral centroid from the audio data.
    
    :param y: Audio time series.
    :param sr: Sampling rate.
    :return: Mean spectral centroid.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(spectral_centroid))

def compute_spectral_centroid_similarity(user_y: np.ndarray, user_sr: int, 
                                         app_y: np.ndarray, app_sr: int) -> float:
    """
    Computes the similarity score based on spectral centroid between user's audio and the reference.
    
    :param user_y: User's audio time series.
    :param user_sr: User's audio sampling rate.
    :param app_y: Reference audio time series.
    :param app_sr: Reference audio sampling rate.
    :return: Similarity score (ranging from 0 to 1).
    """
    # Extracting spectral centroid
    user_centroid = extract_spectral_centroid(user_y, user_sr)
    app_centroid = extract_spectral_centroid(app_y, app_sr)

    # Computing the absolute difference between the centroids
    diff = abs(user_centroid - app_centroid)
    
    # Convert difference to similarity (this is a simplistic transformation, and you can use a more complex one if needed)
    max_possible_diff = max(user_centroid, app_centroid)  # This assumes one of the centroids is the highest possible value
    similarity = 1 - (diff / max_possible_diff)
    
    return float(similarity)



# Spectral Bandwidth
def compute_spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute the spectral bandwidth of an audio signal.
    
    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        np.ndarray: The computed spectral bandwidth.
    """
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]


def spectral_bandwidth_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int, use_padding: bool=True) -> float:
    """Compute the similarity between the spectral bandwidths of two audio signals.
    
    Parameters:
        y1 (np.ndarray): The first audio signal.
        sr1 (int): The sample rate of the first audio signal.
        y2 (np.ndarray): The second audio signal.
        sr2 (int): The sample rate of the second audio signal.

    Returns:
        float: The similarity between the two spectral bandwidths.
    """
    # Compute the spectral bandwidths
    sb1 = compute_spectral_bandwidth(y1, sr1)
    sb2 = compute_spectral_bandwidth(y2, sr2)

    if len(sb1)!=len(sb2):
        # Align the lengths of the two sequences
        if use_padding:
            sb1, sb2 = align_sequences(sb1, sb2)
        else:
            # Resample the audio clips to have the same length
            y1, y2 = resample_to_match(y1, y2)
            
            # Compute the spectral bandwidth for both audios
            sb1 = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)
            sb2 = librosa.feature.spectral_bandwidth(y=y2, sr=sr2)
     
    # Compute the mean absolute difference between the two spectral bandwidth sequences
    # and normalize it to obtain a similarity score between 0 and 1
    
    # mean_difference = np.mean(np.abs(sb1 - sb2))
    # max_possible_difference = np.max([np.max(sb1), np.max(sb2)]) - np.min([np.min(sb1), np.min(sb2)])
    # similarity = 1 - (mean_difference / max_possible_difference)

    similarity = 1 - np.mean(np.abs(sb1 - sb2) / (np.abs(sb1) + np.abs(sb2) + 1e-10))
    return similarity


## Spectral Contras
def spectral_contrast_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int, use_padding: bool = False) -> float:

    """
    Compute the similarity between two audio signals based on their spectral contrasts.

    Parameters:
    - y1 (np.ndarray): The waveform array for the first audio signal.
    - sr1 (int): The sampling rate of the first audio signal.
    - y2 (np.ndarray): The waveform array for the second audio signal.
    - sr2 (int): The sampling rate of the second audio signal.
    - use_padding (bool, optional): Whether to use padding or resampling when aligning audio lengths. Defaults to False (resampling).

    Returns:
    - float: The similarity score based on spectral contrast, where 1 represents maximum similarity and 0 represents minimum similarity.
    """
    
    # Calculate the spectral contrast for both audio signals
    S1 = librosa.feature.spectral_contrast(y=y1, sr=sr1)
    S2 = librosa.feature.spectral_contrast(y=y2, sr=sr2)
    
    # Align the computed spectral contrasts
    if use_padding:
        S1, S2 = align_sequences(S1, S2)
    else:
        y1, y2 = resample_to_match(y1, y2)
        S1 = librosa.feature.spectral_contrast(y=y1, sr=sr1)
        S2 = librosa.feature.spectral_contrast(y=y2, sr=sr2)
    
    # Compute the similarity measure
    mean_difference = np.mean(np.abs(S1 - S2))
    max_possible_difference = np.max([np.max(S1), np.max(S2)]) - np.min([np.min(S1), np.min(S2)])
    similarity = 1 - (mean_difference / max_possible_difference)

    # similarity = 1 - np.mean(np.abs(S1 - S2) / (np.abs(S1) + np.abs(S2) + 1e-10))
    
    return similarity



# Similarity function based on spectral features

def aggregate_spectral_similarity(user_y: np.ndarray, user_sr: int, app_y: np.ndarray, app_sr: int, 
                                  weights: dict = None) -> float:
    """
    Compute an aggregate similarity score based on various spectral features.
    
    Parameters:
    - user_y: Audio time series for the user.
    - user_sr: Sampling rate of user's audio.
    - app_y: Audio time series for the app/reference.
    - app_sr: Sampling rate of app's audio.
    - weights: Dictionary containing weights for each feature. If None, default weights are used.
    
    Returns:
    - float: Aggregate similarity score between 0 (completely dissimilar) and 1 (identical).
    """
    
    # Default weights
    if weights is None:
        weights = {
            'centroid': 0.2,
            'spectrogram': 0.3,
            'bandwidth': 0.2,
            'contrast': 0.15,
            'flatness': 0.15
        }
    
    # Calculate similarities for each feature
    centroid_sim = spectral_centroid_similarity(user_y, user_sr, app_y, app_sr)
    spectrogram_sim = spectrogram_similarity(user_y, user_sr, app_y, app_sr)
    bandwidth_sim = spectral_bandwidth_similarity(user_y, user_sr, app_y, app_sr)
    contrast_sim = spectral_contrast_similarity(user_y, user_sr, app_y, app_sr)
    flatness_sim = spectral_flatness_similarity(user_y, user_sr, app_y, app_sr)
    
    # Aggregate with weights
    aggregate_similarity = (
        weights['centroid'] * centroid_sim +
        weights['spectrogram'] * spectrogram_sim +
        weights['bandwidth'] * bandwidth_sim +
        weights['contrast'] * contrast_sim +
        weights['flatness'] * flatness_sim
    )
    
    return aggregate_similarity











