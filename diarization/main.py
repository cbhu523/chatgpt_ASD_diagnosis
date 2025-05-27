#!/usr/bin/env python
# coding: utf-8
#!pip install pandas tqdm soundfile 

# ───────── Pyannote + Whisper ─────────
#!pip install pyannote.audio openai-whisper
#  HF_TOKEN  ➜  export HF_TOKEN="YOUR_HF_TOKEN"

# ───────── Azure  ─────────
#!pip install azure-cognitiveservices-speech
# 需要 AZURE_SPEECH_KEY & AZURE_SPEECH_REGION

# ───────── Google  ─────────
#!pip install google-cloud-speech
#  GOOGLE_APPLICATION_CREDENTIALS -> JSON

import os, json, argparse, time, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                         --- 3x back-end wrappers ---                        #
# --------------------------------------------------------------------------- #

def transcribe_pyannote(wav_path: Path):
    """
    Return list[dict] with 'speaker' (int from 1) and 'text'.
    Require:
      * HF_TOKEN in env.
      * pyannote.audio >= 3.1
      * openai-whisper
    """
    from pyannote.audio import Pipeline
    import whisper, soundfile as sf

    # --- diarization ---
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set for pyannote.")
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=token
    )
    diar = diar_pipeline(wav_path)

    # --- whisper ASR (full utterance, then we cut) ---
    asr_model = whisper.load_model("base")
    # Whisper expects fp32 PCM wav
    audio, sr = sf.read(wav_path)
    result = asr_model.transcribe(audio, fp16=False, language="en")
    full_text = result["text"]

    # --- align diarization segments with whisper words ---
    speaker_map = {track.label: idx+1 for idx, track in enumerate(diar.tracks())}
    spk_id = list(speaker_map.values())[0] if speaker_map else 1
    return [{"speaker": spk_id, "text": full_text.strip()}]

def transcribe_azure(wav_path: Path):
    """
    Azure Cognitive Services Speech SDK
    Need AZURE_SPEECH_KEY / AZURE_SPEECH_REGION
    """
    import azure.cognitiveservices.speech as speechsdk

    key    = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("Azure env vars not set.")

    speech_cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_cfg.request_word_level_timestamps()
    speech_cfg.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_ProfanityOption,
        value='raw'
    )
    speech_cfg.enable_speaker_diarization()
    audio_input = speechsdk.AudioConfig(filename=str(wav_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_cfg,
                                            audio_config=audio_input)
    done, results = False, []

    def handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            spk = evt.result.speaker_id  # starts from 1
            results.append({"speaker": spk, "text": evt.result.text})

    recognizer.recognized.connect(handler)
    recognizer.session_stopped.connect(lambda evt: setattr(sys.modules[__name__], "done", True))
    recognizer.canceled.connect(lambda evt: setattr(sys.modules[__name__], "done", True))

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)
    recognizer.stop_continuous_recognition()

    return results

def transcribe_google(wav_path: Path):
    """
    Google Cloud Speech-to-Text v2
    Need GOOGLE_APPLICATION_CREDENTIALS
    """
    from google.cloud import speech_v2 as speech

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        diarization_config=speech.SpeakerDiarizationConfig(
            min_speaker_count=2, max_speaker_count=6
        ),
        language_codes=["en-US"],
        model="latest_long"
    )
    with open(wav_path, "rb") as f:
        content = f.read()
    req = speech.RecognizeRequest(
        config=config,
        content=content
    )
    resp = client.recognize(request=req)

    results = []
    for chunk in resp.results:
        alt = chunk.alternatives[0]
        spk = alt.words[0].speaker_label if alt.words else 1
        results.append({"speaker": spk, "text": alt.transcript})
    return results

# --------------------------------------------------------------------------- #
#                     --- helper: assemble speaker text ---                   #
# --------------------------------------------------------------------------- #

def join_segments(segments):
    """
    segments: list[{'speaker': int, 'text': str}]
    → combined string with [SPK_n] prefix per continuous speech block.
    """
    if not segments:
        return ""
    output = []
    prev_spk = None
    for seg in segments:
        spk = seg["speaker"]
        if spk != prev_spk:
            output.append(f"[SPK_{spk}] ")
            prev_spk = spk
        output.append(seg["text"].strip() + " ")
    return "".join(output).strip()

# --------------------------------------------------------------------------- #
#                              --- main ---                                   #
# --------------------------------------------------------------------------- #

def main(args):
    df = pd.read_csv(args.csv)
    assert {"uid", "label"}.issubset({c.lower() for c in df.columns}), \
        "CSV must contain 'uid' and 'label' columns."

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    if args.method == "pyannote":
        transcribe = transcribe_pyannote
    elif args.method == "azure":
        transcribe = transcribe_azure
    elif args.method == "google":
        transcribe = transcribe_google
    else:
        raise ValueError("method must be pyannote | azure | google")

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        uid   = row["uid"]
        label = int(row["label"])
        wav_path = (
            Path(row["wav_path"]) if "wav_path" in row and not pd.isna(row["wav_path"])
            else Path(args.audio_dir) / f"{uid}.wav"
        )
        if not wav_path.exists():
            print(f" Missing audio: {wav_path}", file=sys.stderr)
            continue

        try:
            segments = transcribe(wav_path)
            text = join_segments(segments)
        except Exception as e:
            print(f" {uid} failed: {e}", file=sys.stderr)
            continue

        records.append({"uid": uid, "text": text, "label": label})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output}  (records: {len(records)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="metadata csv with uid,label[,wav_path]")
    p.add_argument("--audio_dir", default=".", help="directory containing .wav files")
    p.add_argument("--method", choices=["pyannote", "azure", "google"], required=True)
    p.add_argument("--output", default="caltech_A4.json")
    args = p.parse_args()
    main(args)

