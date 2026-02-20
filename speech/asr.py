import torch
from transformers import AutoProcessor, AutoModelForCTC
import soundfile as sf

MODEL_NAME = "badrex/w2v-bert-2.0-kikuyu-asr"

class KikuyuASR:
    def __init__(self):
        # Load processor and model from Hugging Face
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCTC.from_pretrained(MODEL_NAME)
            self.model.eval()
            self.model.to("cpu")  # Force CPU usage
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model: {e}")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        Supports .wav, .mp3, .flac. Converts stereo to mono automatically.
        """
        try:
            # Read audio file
            speech, rate = sf.read(audio_path)

            # Convert stereo to mono if necessary
            if len(speech.shape) > 1:
                speech = speech.mean(axis=1)

            # Prepare model inputs
            inputs = self.processor(
                speech,
                sampling_rate=rate,
                return_tensors="pt",
                padding=True
            )

            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Decode predicted IDs
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription

        except FileNotFoundError:
            raise RuntimeError(f"Audio file not found: {audio_path}")
        except Exception as e:
            raise RuntimeError(f"ASR transcription failed: {e}")