import os
import numpy as np
import librosa
from scipy.signal import resample
from sklearn.metrics.pairwise import cosine_similarity
import dotenv


# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())

# Directory where mp3 are stored.
AUDIO_DIR = os.environ.get('AUDIO_DIR')


def load_and_convert_to_mono(audio_file: str) -> np.ndarray:
    """
    Load the raw audio file and convert it to mono.

    Parameters:
        audio_file (str): The path to the raw audio file.

    Returns:
        np.ndarray: The audio data in mono format.
    """
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    return audio, sr


def resample_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    # Resample the audio to the target length using linear interpolation
    resampled_audio = resample(audio, target_length)
    return resampled_audio

def compute_cosine_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    # Compute the cosine similarity between two frames
    similarity = cosine_similarity(frame1.reshape(1, -1), frame2.reshape(1, -1))
    return similarity[0][0]


def get_different_sample_from_same_song(track_id: str):

    filename = os.path.join('.'+ AUDIO_DIR, "000", "000" + track_id + '.mp3')
    x, sr = librosa.load(filename, sr=None, mono=True)
    duration = x.shape[-1] / sr
    
    # compare first two thirds and last two thirds
    third = int(duration*1/3)
    first_two = x[:2*third*sr]
    second_two = x[third*sr:third*sr*3]
    return first_two, second_two, sr
    

def get_resampling_similarity_from_frame(audio1: np.ndarray, audio2: np.ndarray, target_length=1000) -> float:
    
    # Resample audio to a common target length
    audio1_resampled = resample_audio(audio1, target_length)
    audio2_resampled = resample_audio(audio2, target_length)
    
    # Compute cosine similarity between resampled audio frames
    similarity = compute_cosine_similarity(audio1_resampled, audio2_resampled)
    
    return abs(similarity)


def get_resampling_similarity_from_track_id(track_id1: str, track_id2: str, target_length=1000) -> float:

    
    filename1 = os.path.join('.'+ AUDIO_DIR, "000", "000" + track_id1 + '.mp3')
    filename2 =  os.path.join('.'+ AUDIO_DIR, "000", "000" + track_id2 + '.mp3')

    audio1, sr1 = load_and_convert_to_mono(filename1)
    audio2, sr2 = load_and_convert_to_mono(filename2)
    assert sr1==sr2
    
    return get_resampling_similarity_from_frame(audio1, audio2, target_length=target_length)


def get_similarity_within_same_song_by_resampling(track_id: str) -> float:

    first_two, second_two, sr = get_different_sample_from_same_song(track_id)
    similarity = get_resampling_similarity_from_frame(first_two, second_two)
    return similarity