import librosa
import numpy as np
import noisereduce as nr
from typing import Tuple, Callable


def detect_start_time(y: np.ndarray, sr: int, energy_threshold: float = 0.01) -> float:
    """
    Determines the time when actual playing starts in a user's audio recording.

    Reasoning:
    User might have some silence or ambient noise before they start playing. 
    This function detects the start of the actual playing based on the energy 
    of the audio.
    """
    frame_length = int(sr * 0.02)
    hop_length = frame_length // 2
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    frames_exceeding_threshold = np.where(energy > energy_threshold)[0]
    if len(frames_exceeding_threshold) > 0:
        start_frame = frames_exceeding_threshold[0]
        return start_frame * hop_length / sr
    return 0


def noise_reduction(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Reduces noise in the user's audio recording.

    Reasoning:
    User recordings might have ambient noises. Reducing noise can lead to 
    better evaluation results.
    """
    return nr.reduce_noise(y=y, sr=sr)


def calculate_pitch(y: np.ndarray, sr: int) -> list:
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    main_pitches = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        main_pitches.append(pitches[index, i] if magnitudes[index, i] > np.max(magnitudes) * 0.1 else 0) # include threshold
    return main_pitches


def calculate_onset_times(y: np.ndarray, sr: int) -> list:
    """
    Determines the onset times in the audio. 
    For user's audio, adjusts the onset times based on the detected start time.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    start_time = detect_start_time(y, sr)
    adjusted_onset_times = [time - start_time for time in onset_times]
    return adjusted_onset_times


def score_accuracy(user_pitches: list, app_pitches: list) -> float:
    """
    Compares the pitches from user's audio and app's audio to calculate accuracy.
    """
    min_len = min(len(user_pitches), len(app_pitches))
    correct_notes = sum(1 for i in range(min_len) if abs(user_pitches[i] - app_pitches[i]) < 10)
    return (correct_notes / min_len) * 100


def score_timing(user_onsets: list, app_onsets: list) -> float:
    """
    Evaluates how well the user's timing matches the app's audio.
    A lower difference means the user's timing is closer to the app's, which is better.
    """
    timing_diff = [abs(user_onsets[i] - app_onsets[i]) for i in range(min(len(user_onsets), len(app_onsets)))]
    avg_diff = np.mean(timing_diff)
    return max(0, 100 - avg_diff * 20)


def score_fluency(user_y: np.ndarray, sr: int, penalization_factor: int=1) -> float:
    """
    Assesses the fluency of the user's playing. 
    Fewer onset times might indicate fewer breaks or hesitations.
    """
    fluency = 100 - len(calculate_onset_times(user_y, sr)) * penalization_factor
    return max(0, fluency)


def basic_feat_sim(path_to_user_audio: str, path_to_app_audio: str) -> float:
    """
    Main evaluation function that computes a score for user's performance based on app's audio.
    """
    user_y, user_sr = librosa.load(path_to_user_audio, sr=None)
    app_y, app_sr = librosa.load(path_to_app_audio, sr=None)

    target_sample_rate = 44100
    if user_sr != target_sample_rate:
        user_y = librosa.resample(user_y, user_sr, target_sample_rate)
    if app_sr != target_sample_rate:
        app_y = librosa.resample(app_y, app_sr, target_sample_rate)

    user_y = noise_reduction(user_y, target_sample_rate)

    user_pitches = calculate_pitch(user_y, target_sample_rate)
    app_pitches = calculate_pitch(app_y, target_sample_rate)
    accuracy = score_accuracy(user_pitches, app_pitches)

    user_onsets = calculate_onset_times(user_y, target_sample_rate)
    # We don't adjust onset times for the app's audio as it starts from t=0.
    app_onsets = librosa.frames_to_time(librosa.onset.onset_detect(y=app_y, sr=target_sample_rate), sr=target_sample_rate)
    timing = score_timing(user_onsets, app_onsets)

    fluency = score_fluency(user_y, target_sample_rate)

    return (accuracy + timing + fluency) / 3




# Test
if __name__ == '__main__':

    audio_path = '../data/fma_small/000/000'
    track_id1 = "002.mp3"
    track_id2 = "368.mp3"
    filename1 = os.path.join(audio_path+track_id1)
    filename2 =  os.path.join(audio_path+track_id2)
    
    final_score = basic_feat_sim(filename1, filename1)
    print(f"Final Score: {final_score:.2f}%")




