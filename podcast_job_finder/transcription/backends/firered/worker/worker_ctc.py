from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-not-found]

from worker_features import FeatureExtractor  # type: ignore[import-not-found]


BLANK_TOKEN = "<blank>"
ENCODER_FRAME_SHIFT_MS = 40


class TokenDictionary:
    def __init__(self, path: Path) -> None:
        self._id_to_token: dict[int, str] = {}
        self._token_to_id: dict[str, int] = {}
        with path.open(encoding="utf-8") as file_obj:
            for line_index, line in enumerate(file_obj):
                fields = line.strip().split()
                if not fields:
                    continue
                token = fields[0]
                token_id = (
                    int(fields[1])
                    if len(fields) >= 2 and fields[1].isdigit()
                    else line_index
                )
                self._id_to_token[token_id] = token
                self._token_to_id[token] = token_id

    def id(self, token: str) -> int:
        return self._token_to_id[token]

    def text(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")


@dataclass(slots=True, frozen=True)
class EncodedAudio:
    outputs: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray

    @property
    def valid_frame_count(self) -> int:
        return int(self.lengths[0])


class FireRedCtcModel:
    def __init__(
        self,
        model_dir: Path,
        *,
        provider: str,
        intra_op_threads: int,
    ) -> None:
        self.tokens = TokenDictionary(model_dir / "tokens.txt")
        self.blank_id = self.tokens.id(BLANK_TOKEN)
        self._features = FeatureExtractor(model_dir / "cmvn.ark")
        self._encoder = load_onnx_session(
            model_dir / "encoder.int8.onnx",
            provider=provider,
            intra_op_threads=intra_op_threads,
        )
        self._ctc = load_onnx_session(
            model_dir / "ctc.int8.onnx",
            provider=provider,
            intra_op_threads=intra_op_threads,
        )

    def encode(self, audio_path: Path) -> EncodedAudio:
        features, feature_lengths = self._features.extract(audio_path)
        encoder_outputs, encoder_lengths, encoder_mask = self._encoder.run(
            ["output", "output_lengths", "mask"],
            {"input": features, "input_lengths": feature_lengths},
        )
        return EncodedAudio(
            outputs=encoder_outputs,
            lengths=encoder_lengths,
            mask=encoder_mask,
        )

    def infer_logits(self, audio_path: Path) -> np.ndarray:
        return self.build_logits(self.encode(audio_path))

    def build_logits(self, encoded_audio: EncodedAudio) -> np.ndarray:
        logits = self._ctc.run(
            ["logits"],
            {"encoder_outputs": encoded_audio.outputs.astype(np.float32)},
        )[0]
        return logits[0, : encoded_audio.valid_frame_count]


def load_onnx_session(
    path: Path,
    *,
    provider: str,
    intra_op_threads: int,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_op_threads
    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), options, providers=providers)
