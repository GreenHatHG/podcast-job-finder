from __future__ import annotations

# This script runs in the dedicated FireRed environment and imports sibling modules.
# pylint: disable=import-error

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]
from transformers import BertModel, BertTokenizer  # type: ignore[import-not-found]

from worker_asr import AlignedToken  # type: ignore[import-not-found]


SPACE_TAG = " "
SPACE_TOKEN = "<space>"


@dataclass(slots=True, frozen=True)
class PunctuatedSentence:
    text: str
    start_ms: int
    end_ms: int


class _CheckpointArgs(Protocol):
    classifier_dropout: float
    odim: int
    cls_id: int


class _FireRedPuncBert(nn.Module):
    def __init__(self, checkpoint_args: _CheckpointArgs, bert: BertModel) -> None:
        super().__init__()
        self.bert = bert
        self.bert.pooler = None
        self.dropout = nn.Dropout(float(checkpoint_args.classifier_dropout))
        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            int(checkpoint_args.odim),
        )
        self.cls_id = int(checkpoint_args.cls_id)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        cls_tokens = token_ids.new_full((token_ids.size(0), 1), self.cls_id)
        model_inputs = torch.cat((cls_tokens, token_ids), dim=1)
        attention_mask = torch.ones_like(model_inputs)
        sequence_output = self.bert(
            model_inputs,
            attention_mask=attention_mask,
        )[0][:, 1:]
        return self.classifier(self.dropout(sequence_output))


class FireRedPunctuation:
    def __init__(self, model_dir: Path) -> None:
        bert_dir = model_dir / "chinese-lert-base"
        package = torch.load(
            model_dir / "model.pth.tar",
            map_location="cpu",
            weights_only=False,
        )
        bert = BertModel.from_pretrained(str(bert_dir))
        checkpoint_args = cast(_CheckpointArgs, package["args"])
        self._model = _FireRedPuncBert(checkpoint_args, bert)
        self._model.load_state_dict(package["model_state_dict"], strict=False)
        self._model.eval()
        self._tokenizer = BertTokenizer.from_pretrained(str(bert_dir))
        self._tags = _load_output_tags(model_dir / "out_dict")

    @torch.no_grad()
    def punctuate(
        self,
        tokens: list[AlignedToken],
    ) -> tuple[str, list[PunctuatedSentence]]:
        if not tokens:
            return "", []
        input_ids, split_counts = self._build_inputs(tokens)
        logits = self._model(torch.tensor([input_ids], dtype=torch.long))[0]
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        tags = _merge_subtoken_tags(predictions, split_counts, self._tags)
        sentences = _build_sentences(tokens, tags)
        return "".join(sentence.text for sentence in sentences), sentences

    def _build_inputs(
        self,
        tokens: list[AlignedToken],
    ) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        split_counts = []
        for token in tokens:
            subtokens = self._tokenizer.tokenize(token.text)
            if not subtokens:
                subtokens = [self._tokenizer.unk_token]
            subtoken_ids = self._tokenizer.convert_tokens_to_ids(subtokens)
            input_ids.extend(int(token_id) for token_id in subtoken_ids)
            split_counts.append(len(subtoken_ids))
        return input_ids, split_counts


def _load_output_tags(path: Path) -> list[str]:
    tags = []
    with path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            fields = line.strip().split()
            tag = fields[0] if fields else SPACE_TAG
            tags.append(SPACE_TAG if tag == SPACE_TOKEN else tag)
    return tags


def _merge_subtoken_tags(
    predictions: list[int],
    split_counts: list[int],
    tags: list[str],
) -> list[str]:
    merged_tags = []
    offset = 0
    for split_count in split_counts:
        prediction_index = predictions[offset + split_count - 1]
        merged_tags.append(tags[prediction_index])
        offset += split_count
    return merged_tags


def _build_sentences(
    tokens: list[AlignedToken],
    tags: list[str],
) -> list[PunctuatedSentence]:
    sentences = []
    text_parts: list[str] = []
    sentence_start_ms: int | None = None
    sentence_end_ms = tokens[0].end_ms
    for token, tag in zip(tokens, tags):
        if sentence_start_ms is None:
            sentence_start_ms = token.start_ms
        text_parts.append(token.text)
        sentence_end_ms = token.end_ms
        if tag == SPACE_TAG:
            continue
        text_parts.append(tag)
        sentences.append(
            PunctuatedSentence(
                text="".join(text_parts),
                start_ms=sentence_start_ms,
                end_ms=sentence_end_ms,
            )
        )
        text_parts = []
        sentence_start_ms = None
    if text_parts:
        if sentence_start_ms is None:
            raise ValueError("FireRedPunc 句子起始时间缺失。")
        sentences.append(
            PunctuatedSentence(
                text="".join(text_parts),
                start_ms=sentence_start_ms,
                end_ms=sentence_end_ms,
            )
        )
    return sentences
