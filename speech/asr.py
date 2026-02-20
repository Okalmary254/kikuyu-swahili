
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoProcessor, AutoModelForCTC
import soundfile as sf

MODEL_NAME = "badrex/w2v-bert-2.0-kikuyu-asr"

class KikuyuASR:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCTC.from_pretrained(MODEL_NAME)
        self.model.eval()

    def transcribe(self, audio_path: str) -> str:
        speech, rate = sf.read(audio_path)

        inputs = self.processor(
            speech,
            sampling_rate=rate,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription
