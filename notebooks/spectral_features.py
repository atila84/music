import numpy as np
import librosa

def align_sequences(seq1: np.ndarray, seq2: np.ndarray) -> (np.ndarray, np.ndarray):
    """Align two sequences by padding the shorter sequence with zeros."""
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
    

def spectrogram_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int, use_cosine: bool=False) -> float:
    """Compute the similarity between two audio signals based on their spectrograms."""
    S1 = np.abs(librosa.stft(y1))
    S2 = np.abs(librosa.stft(y2))
    S1, S2 = align_sequences(S1, S2)
    if use_cosine:
        spec1_vec = S1.flatten()
        spec2_vec = S2.flatten()

        # Ensure that both vectors have the same shape
        min_len = min(len(spec1_vec), len(spec2_vec))
        spec1_vec = spec1_vec[:min_len]
        spec2_vec = spec2_vec[:min_len]
        
        # Compute cosine similarity
        return cosine_similarity(spec1_vec.reshape(1, -1), spec2_vec.reshape(1, -1))

    mean_difference = np.mean(np.abs(S1 - S2))
    max_possible_difference = np.max([np.max(S1), np.max(S2)]) - np.min([np.min(S1), np.min(S2)])
    similarity = 1 - (mean_difference / max_possible_difference)
    return similarity

def spectral_centroid_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int) -> float:
    """Compute the similarity between two audio signals based on their spectral centroids."""
    sc1 = librosa.feature.spectral_centroid(y=y1, sr=sr1)[0]
    sc2 = librosa.feature.spectral_centroid(y=y2, sr=sr2)[0]
    sc1, sc2 = align_sequences(sc1, sc2)
    sim = 1 - np.mean(np.abs(sc1 - sc2) / (np.abs(sc1) + np.abs(sc2) + 1e-10))
    return sim

def spectral_bandwidth_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int) -> float:
    """Compute the similarity between two audio signals based on their spectral bandwidths."""
    sb1 = librosa.feature.spectral_bandwidth(y=y1, sr=sr1)[0]
    sb2 = librosa.feature.spectral_bandwidth(y=y2, sr=sr2)[0]
    sb1, sb2 = align_sequences(sb1, sb2)
    sim = 1 - np.mean(np.abs(sb1 - sb2) / (np.abs(sb1) + np.abs(sb2) + 1e-10))
    return sim

def spectral_contrast_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int) -> float:
    """Compute the similarity between two audio signals based on their spectral contrasts."""
    S1 = np.abs(librosa.stft(y1))
    S2 = np.abs(librosa.stft(y2))
    contrast1 = librosa.feature.spectral_contrast(S=S1, sr=sr1).flatten()
    contrast2 = librosa.feature.spectral_contrast(S=S2, sr=sr2).flatten()
    contrast1, contrast2 = align_sequences(contrast1, contrast2)
    mean_difference = np.mean(np.abs(contrast1 - contrast2))
    max_possible_difference = np.max([np.max(contrast1), np.max(contrast2)]) - np.min([np.min(contrast1), np.min(contrast2)])
    similarity = 1 - (mean_difference / max_possible_difference)
    return similarity

def spectral_flatness_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int) -> float:
    """Compute the similarity between two audio signals based on their spectral flatness."""
    sf1 = librosa.feature.spectral_flatness(y=y1)[0]
    sf2 = librosa.feature.spectral_flatness(y=y2)[0]
    sf1, sf2 = align_sequences(sf1, sf2)
    mean_difference = np.mean(np.abs(sf1 - sf2))
    max_possible_difference = np.max([np.max(sf1), np.max(sf2)]) - np.min([np.min(sf1), np.min(sf2)])
    similarity = 1 - (mean_difference / max_possible_difference)
    return similarity

def aggregate_similarity(y1: np.ndarray, sr1: int, y2: np.ndarray, sr2: int, weights: dict = None) -> float:
    """Compute an aggregate similarity score between two audio signals based on various spectral features."""
    if weights is None:
        weights = {
            'spectrogram': 0.2,
            'centroid': 0.2,
            'sb': 0.2,
            'sc': 0.2,
            'sf': 0.2
        }

    spectrogram_sim = spectrogram_similarity(y1, sr1, y2, sr2)
    centroid_sim = spectral_centroid_similarity(y1, sr1, y2, sr2)
    sb_sim = spectral_bandwidth_similarity(y1, sr1, y2, sr2)
    sc_sim = spectral_contrast_similarity(y1, sr1, y2, sr2)
    sf_sim = spectral_flatness_similarity(y1, sr1, y2, sr2)

    aggregate_similarity = (weights['spectrogram'] * spectrogram_sim +
                            weights['centroid'] * centroid_sim +
                            weights['sb'] * sb_sim +
                            weights['sc'] * sc_sim +
                            weights['sf'] * sf_sim)
    return aggregate_similarity
