import numpy as np
import librosa

def load_and_convert_to_mono(audio_file: str) -> np.ndarray:
    """
    Load the raw audio file and convert it to mono.

    Parameters:
        audio_file (str): The path to the raw audio file.

    Returns:
        np.ndarray: The audio data in mono format.
    """
    audio, _ = librosa.load(audio_file, sr=None, mono=True)
    return audio


def compute_distance(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two frames.

    Parameters:
        frame1 (np.ndarray): The first audio frame.
        frame2 (np.ndarray): The second audio frame.

    Returns:
        float: The Euclidean distance between the two frames.
    """
    return np.sqrt(np.sum((frame1 - frame2) ** 2))


def compute_smoothness(wp: np.ndarray) -> float:
    """
    Compute the smoothness of the alignment path.

    Parameters:
        wp (np.ndarray): The alignment path in the format [[i1, j1], [i2, j2], ...].

    Returns:
        float: The smoothness value indicating the consistency of alignment.
    """
    deltas = np.diff(wp, axis=0)
    i_deltas = deltas[:, 0]
    j_deltas = deltas[:, 1]
    i_smoothness = np.std(i_deltas)
    j_smoothness = np.std(j_deltas)
    overall_smoothness = (i_smoothness + j_smoothness) / 2
    return overall_smoothness


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize the audio frame to have values in the range [0, 1].

    Parameters:
        audio (np.ndarray): The audio frame.

    Returns:
        np.ndarray: The normalized audio frame.
    """
    return (audio - np.min(audio)) / (np.max(audio) - np.min(audio))


def calculate_similarity_from_path(audio_file1: str, audio_file2: str) -> float:
    """
    Calculate the similarity of two audio frames using Dynamic Time Warping (DTW).

    Parameters:
        audio_file1 (str): The path to the first raw audio file.
        audio_file2 (str): The path to the second raw audio file.

    Returns:
        float: The similarity score between the two audio frames.
    """
    # Load and convert raw audio to mono
    audio1 = load_and_convert_to_mono(audio_file1)
    audio2 = load_and_convert_to_mono(audio_file2)

    # Normalize the audio frames
    audio1_normalized = normalize_audio(audio1)
    audio2_normalized = normalize_audio(audio2)

    # Compute the distance between frames using Euclidean distance
    distance = compute_distance(audio1_normalized, audio2_normalized)

    # Prepare the sequences for DTW
    X = np.atleast_2d(audio1_normalized)
    Y = np.atleast_2d(audio2_normalized)

    # Compute the DTW cost matrix and optimal alignment path
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)

    # Compute the smoothness of the alignment path
    smoothness = compute_smoothness(wp)

    # Compute the length of the alignment path
    path_length = len(wp)

    # Calculate the overall similarity score (you can adjust weights based on preference)
    similarity_score = 0.5 * distance + 0.3 * (1 / smoothness) + 0.2 * (1 / path_length)

    return similarity_score


def calculate_similarity_from_audio(audio_file1: str, audio_file2: str) -> float:
    """
    Calculate the similarity of two audio frames using Dynamic Time Warping (DTW).

    Parameters:
        audio_file1 (str): The path to the first raw audio file.
        audio_file2 (str): The path to the second raw audio file.

    Returns:
        float: The similarity score between the two audio frames.
    """
    # Load and convert raw audio to mono
    audio1 = load_and_convert_to_mono(audio_file1)
    audio2 = load_and_convert_to_mono(audio_file2)

    # Normalize the audio frames
    audio1_normalized = normalize_audio(audio1)
    audio2_normalized = normalize_audio(audio2)

    # Compute the distance between frames using Euclidean distance
    distance = compute_distance(audio1_normalized, audio2_normalized)

    # Prepare the sequences for DTW
    X = np.atleast_2d(audio1_normalized)
    Y = np.atleast_2d(audio2_normalized)

    # Compute the DTW cost matrix and optimal alignment path
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)

    # Compute the smoothness of the alignment path
    smoothness = compute_smoothness(wp)

    # Compute the length of the alignment path
    path_length = len(wp)

    # Calculate the overall similarity score (you can adjust weights based on preference)
    similarity_score = 0.5 * distance + 0.3 * (1 / smoothness) + 0.2 * (1 / path_length)

    return similarity_score


# Example usage:
# audio_file1 = "path/to/audio1.wav"
# audio_file2 = "path/to/audio2.wav"

# similarity_score = calculate_similarity(audio_file1, audio_file2)
# print("Similarity Score:", similarity_score)
