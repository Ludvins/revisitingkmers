import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as util_data
import transformers
from sklearn.preprocessing import normalize
from embedders.base import BaseEmbedder, EmbeddingResult
from utils.progress import pbar


class LLMEmbedder(BaseEmbedder):
    """Embedding via pretrained DNA language models (DNABERT-2, HyenaDNA, NT, DNABERT-S).

    Computes mean-pooled hidden states from a HuggingFace transformer model.
    """

    def __init__(self, model_name_or_path: str, model_max_length: int = 5000,
                 batch_size: int = 10):
        self._model_name_or_path = model_name_or_path
        self._model_max_length = model_max_length
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            cache_dir=None,
            model_max_length=self._model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        is_nt = "nucleotide-transformer" in self._model_name_or_path
        if is_nt:
            self._model = transformers.AutoModelForMaskedLM.from_pretrained(
                self._model_name_or_path, trust_remote_code=True,
            )
        else:
            self._model = transformers.AutoModel.from_pretrained(
                self._model_name_or_path, trust_remote_code=True,
            )

        n_gpu = torch.cuda.device_count()
        if n_gpu >= 1:
            self._model = nn.DataParallel(self._model)
            self._model.to("cuda")
        else:
            self._model.to("cpu")

    def embed(self, sequences: list[str]) -> EmbeddingResult:
        self._load_model()

        is_hyenadna = "hyenadna" in self._model_name_or_path
        is_nt = "nucleotide-transformer" in self._model_name_or_path

        # Sort by length for efficient batching
        lengths = [len(seq) for seq in sequences]
        idx = np.argsort(lengths)
        sorted_seqs = [sequences[i] for i in idx]

        n_gpu = torch.cuda.device_count()
        n_cpu = 0 if n_gpu >= 1 else 1
        device = "cuda" if n_gpu >= 1 else "cpu"

        loader = util_data.DataLoader(
            sorted_seqs,
            batch_size=self._batch_size * (n_gpu + n_cpu),
            shuffle=False,
            num_workers=2 * (n_gpu + n_cpu),
        )

        all_embeddings = None
        for j, batch in enumerate(pbar(loader, desc="Embedding sequences", unit="batch")):
            with torch.inference_mode():
                token_feat = self._tokenizer.batch_encode_plus(
                    batch,
                    max_length=self._model_max_length,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )
                input_ids = token_feat["input_ids"]
                if not is_hyenadna:
                    attention_mask = token_feat["attention_mask"]

                if n_gpu:
                    input_ids = input_ids.cuda()
                    if not is_hyenadna:
                        attention_mask = attention_mask.cuda()

                if is_hyenadna:
                    model_output = self._model.forward(input_ids=input_ids)[0].detach().cpu()
                    attention_mask = torch.ones(
                        size=(model_output.shape[0], model_output.shape[1], 1), device="cpu"
                    )
                else:
                    model_output = self._model.forward(
                        input_ids=input_ids, attention_mask=attention_mask
                    )[0].detach().cpu()
                    attention_mask = attention_mask.unsqueeze(-1).detach().cpu()

                embedding = (
                    torch.sum(model_output * attention_mask, dim=1)
                    / torch.sum(attention_mask, dim=1)
                )

                if all_embeddings is None:
                    all_embeddings = embedding
                else:
                    all_embeddings = torch.cat((all_embeddings, embedding), dim=0)

        embeddings = np.array(all_embeddings.detach().cpu())
        # Restore original ordering
        embeddings = embeddings[np.argsort(idx)]
        embeddings = normalize(embeddings, norm="l2")
        return EmbeddingResult(mean=embeddings)

    def save(self, path: str) -> None:
        pass  # Uses pretrained models, nothing to save

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "LLMEmbedder":
        return cls(model_name_or_path=path, **kwargs)

    @property
    def default_metric(self) -> str:
        return "dot"


# Pre-configured LLM embedder factories
KNOWN_LLM_MODELS = {
    "hyenadna": {
        "model_name_or_path": "LongSafari/hyenadna-medium-450k-seqlen-hf",
        "model_max_length": 20000,
    },
    "dnabert2": {
        "model_name_or_path": "zhihan1996/DNABERT-2-117M",
        "model_max_length": 5000,
    },
    "nt": {
        "model_name_or_path": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "model_max_length": 2048,
    },
}
