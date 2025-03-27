import gc
import hashlib
import json
import logging
import os
import shlex
import subprocess
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from zipfile import ZipFile

import librosa
import numpy as np
import requests
import torch
from fairseq import checkpoint_utils
from mdx import run_mdx
from pedalboard.io import AudioFile
from pydub import AudioSegment
from scipy.io import wavfile
from multiprocessing import cpu_count

import yt_dlp
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, device: str, is_half: bool):
        """
        Device configuration for inference.

        Args:
            device (str): Device identifier (e.g., "cuda:0", "cpu", "mps").
            is_half (bool): Whether to use half precision.
        """
        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory / (1024**3) + 0.4
            )
        elif torch.backends.mps.is_available():
            logger.info("No supported Nvidia card found, using MPS for inference.")
            self.device = "mps"
        else:
            logger.info("No supported GPU found, using CPU for inference.")
            self.device = "cpu"
            self.is_half = True

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device: str, is_half: bool, model_path: str):
    """
    Loads the Hubert model.

    Args:
        device (str): Device to load the model on.
        is_half (bool): Whether to use half precision.
        model_path (str): Path to the model checkpoint.

    Returns:
        The loaded Hubert model.
    """
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], suffix=''
    )
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    return hubert


def get_vc(device: str, is_half: bool, config: Config, model_path: str):
    """
    Loads a voice conversion model.

    Args:
        device (str): Device identifier.
        is_half (bool): Whether to use half precision.
        config (Config): Configuration object.
        model_path (str): Path to the model checkpoint.

    Returns:
        A tuple: (checkpoint, version, network model, target sample rate, VC pipeline object)
    """
    cpt = torch.load(model_path, map_location="cpu")
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(
            f"Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead."
        )

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g = net_g.eval().to(device)
    net_g = net_g.half() if is_half else net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def svc_infer(
    index_path: str,
    index_rate: float,
    input_path: str,
    output_path: str,
    pitch_change: float,
    f0_method: str,
    cpt: dict,
    version: str,
    net_g,
    filter_radius: int,
    tgt_sr: int,
    rms_mix_rate: float,
    protect: float,
    crepe_hop_length: int,
    vc,
    hubert_model,
) -> None:
    """
    Performs inference on a given audio file and saves the output.

    Args:
        index_path (str): Path to the index file.
        index_rate (float): Index rate.
        input_path (str): Path to the input audio file.
        output_path (str): Path where the output audio will be saved.
        pitch_change (float): Pitch change value.
        f0_method (str): Fundamental frequency extraction method.
        cpt (dict): Checkpoint dictionary.
        version (str): Model version.
        net_g: Network model.
        filter_radius (int): Filter radius.
        tgt_sr (int): Target sample rate.
        rms_mix_rate (float): RMS mix rate.
        protect (float): Protection value.
        crepe_hop_length (int): Hop length for CREPE.
        vc: Voice conversion pipeline.
        hubert_model: Hubert model.
    """
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get("f0", 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        input_path,
        times,
        pitch_change,
        f0_method,
        index_path,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        0,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
    )
    logger.info("Successfully converted! Saving the output file...")
    wavfile.write(output_path, tgt_sr, audio_opt)


def get_model_path(folder: str) -> tuple:
    """
    Searches for model (.pth) and index (.index) files in the given folder.

    Args:
        folder (str): Folder path to search.

    Returns:
        Tuple of (svc_model_path, svc_index_path).
    """
    svc_model_path = svc_index_path = None
    folder_path = Path(folder)
    for filee in folder_path.iterdir():
        if filee.suffix == ".index":
            svc_index_path = str(filee)
        elif filee.suffix == ".pth":
            svc_model_path = str(filee)
    return svc_model_path, svc_index_path


def extract(filename: str, output_path: str = "") -> None:
    """
    Extracts a ZIP file to the specified output directory.

    Args:
        filename (str): Path to the ZIP file.
        output_path (str): Output directory path.
    """
    with ZipFile(filename, "r") as zObject:
        zObject.extractall(path=output_path)


def down(url: str, output_file: str) -> None:
    """
    Downloads content from a URL and saves it to a file.

    Args:
        url (str): URL to download from.
        output_file (str): Path to save the downloaded file.
    """
    r = requests.get(url)
    r.raise_for_status()
    with open(output_file, "wb") as f:
        f.write(r.content)


def convert_to_stereo(audio_path: str) -> str:
    """
    Converts an audio file to stereo if it is mono.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        Path to the stereo audio file.
    """
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    # If wave[0] is not an ndarray then the file is mono
    if not isinstance(wave[0], np.ndarray):
        stereo_path = Path(audio_path).with_name(f"{Path(audio_path).stem}_stereo.wav")
        command = shlex.split(
            f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
        )
        subprocess.run(command, check=True)
        return str(stereo_path)
    else:
        return audio_path


def get_hash(filepath: str) -> str:
    """
    Computes a blake2b hash of the file.

    Args:
        filepath (str): Path to the file.

    Returns:
        First 11 characters of the hexadecimal digest.
    """
    file_hash = hashlib.blake2b()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]


def preprocess_song(
    song_input: str,
    mdx_model_params: dict,
    song_id: str,
    mdxnet_models_dir: str = "mdxnet_models",
    output_dir: str = "song_outputs",
    input_type: str = "yt",
) -> tuple:
    """
    Preprocesses a song by downloading, converting to stereo, and separating vocals/instrumentals.

    Args:
        song_input (str): URL or local path to the song.
        mdx_model_params (dict): Parameters for the MDX model.
        song_id (str): Unique identifier for the song.
        mdxnet_models_dir (str): Directory for MDXNet models.
        output_dir (str): Directory to store outputs.
        input_type (str): Either 'yt' for YouTube or 'local'.

    Returns:
        Tuple containing paths to the original song, vocals, instrumentals, main vocals, and backup vocals.
    """
    keep_orig = False
    if input_type == "yt":
        logger.info("Downloading song from YouTube...")
        song_link = song_input.split("&")[0]
        orig_song_path = yt_download(song_link, output_dir)
    elif input_type == "local":
        orig_song_path = song_input
        keep_orig = True
    else:
        raise ValueError("Invalid input_type. Use 'yt' or 'local'.")

    song_output_dir = Path(output_dir) / song_id
    song_output_dir.mkdir(parents=True, exist_ok=True)

    vocals_wav = f"{Path(orig_song_path).stem}_Vocals.wav"
    instr_wav = f"{Path(orig_song_path).stem}_Instrumental.wav"
    vocals_main_wav = f"{Path(orig_song_path).stem}_Vocals_Main.wav"
    vocals_backup_wav = f"{Path(orig_song_path).stem}_Vocals_Backup.wav"

    vocals_path = str(song_output_dir / vocals_wav)
    instrumentals_path = str(song_output_dir / instr_wav)
    main_vocals_path = str(song_output_dir / vocals_main_wav)
    backup_vocals_path = str(song_output_dir / vocals_backup_wav)

    if not (Path(vocals_path).exists() and Path(instrumentals_path).exists()):
        orig_song_path = convert_to_stereo(orig_song_path)
        logger.info("Separating Vocals from Instrumental...")
        vocals_path, instrumentals_path = run_mdx(
            mdx_model_params,
            str(song_output_dir),
            str(Path(mdxnet_models_dir) / "UVR-MDX-NET-Voc_FT.onnx"),
            orig_song_path,
            denoise=True,
            keep_orig=keep_orig,
        )
    if not (Path(main_vocals_path).exists() and Path(backup_vocals_path).exists()):
        logger.info("Separating Main Vocals from Backup Vocals...")
        backup_vocals_path, main_vocals_path = run_mdx(
            mdx_model_params,
            str(song_output_dir),
            str(Path(mdxnet_models_dir) / "UVR_MDXNET_KARA_2.onnx"),
            vocals_path,
            suffix="Backup",
            invert_suffix="Main",
            denoise=True,
        )
    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path


def combine_audio(
    audio_paths: list,
    output_path: str,
    main_gain: float = 0,
    backup_gain: float = 0,
    inst_gain: float = 0,
    output_format: str = "mp3",
) -> None:
    """
    Combines main vocals, backup vocals, and instrumentals into a single audio file.

    Args:
        audio_paths (list): List of three audio paths [main_vocals, backup_vocals, instrumentals].
        output_path (str): Path to save the combined audio.
        main_gain (float): Gain adjustment for main vocals.
        backup_gain (float): Gain adjustment for backup vocals.
        inst_gain (float): Gain adjustment for instrumentals.
        output_format (str): Format for output file.
    """
    main_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    combined = main_audio.overlay(backup_audio).overlay(instrumental_audio)
    combined.export(output_path, format=output_format)


def yt_download(url: str, output_dir: str) -> str:
    """
    Downloads audio from a YouTube URL and converts it to WAV.

    Args:
        url (str): YouTube URL.
        output_dir (str): Directory where the file will be saved.

    Returns:
        Absolute path to the downloaded WAV file.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "32",
            }
        ],
        "outtmpl": str(output_dir_path / "%(title)s.%(ext)s"),
        "postprocessor_args": ["-acodec", "pcm_f32le"],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info["title"]
        ydl.download([url])
        file_path = output_dir_path / f"{video_title}.wav"
        return str(file_path)


def get_youtube_video_id(url: str, ignore_playlist: bool = True) -> str:
    """
    Extracts the video or playlist ID from a YouTube URL.

    Args:
        url (str): YouTube URL.
        ignore_playlist (bool): Whether to ignore playlist IDs.

    Returns:
        The extracted video ID or None if invalid.
    """
    query = urlparse(url)
    if query.hostname == "youtu.be":
        if query.path[1:] == "watch":
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com"}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)["list"][0]
        if query.path == "/watch":
            return parse_qs(query.query)["v"][0]
        if query.path.startswith("/watch/"):
            return query.path.split("/")[1]
        if query.path.startswith("/embed/"):
            return query.path.split("/")[2]
        if query.path.startswith("/v/"):
            return query.path.split("/")[2]

    return None


# Exported symbols
__all__ = [
    "Config",
    "load_hubert",
    "get_vc",
    "svc_infer",
    "get_model_path",
    "extract",
    "down",
    "convert_to_stereo",
    "get_hash",
    "preprocess_song",
    "combine_audio",
    "yt_download",
    "get_youtube_video_id",
]

