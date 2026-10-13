"""Frozen Genos-m sequence embeddings for the SemiBin clustering pipeline."""

from functools import lru_cache
import re

import numpy as np
import torch


class GenosMEmbedder:
    """Pool Genos-m token states over windows, then over each contig.

    The pooling within a window follows the Genos-m example: an attention-mask
    weighted mean of ``last_hidden_state``. Window means are weighted by the
    number of nucleotides so the end of a long contig is never discarded.
    """

    def __init__(self, model_name, window_size, batch_size, device):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'Genos-m requires the optional dependencies: '
                'pip install "SemiBin[genos-m]"'
            ) from exc

        self.window_size = window_size
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        if device == 'cuda':
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
            device_map = 'auto'
        else:
            dtype = torch.float32
            device_map = None

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.model.config.use_cache = False
        if device_map is None:
            self.model.to(device)

    def embed_contigs(self, contig_ids, contig_dict, logger):
        vectors = []
        ambiguous_bases = 0
        for index, contig_id in enumerate(contig_ids, 1):
            if contig_id not in contig_dict:
                raise ValueError(
                    f'Contig {contig_id!r} in data.csv is missing from the FASTA file')
            sequence = contig_dict[contig_id].upper()
            if not sequence:
                raise ValueError(f'Contig {contig_id!r} has an empty sequence')
            sequence, n_ambiguous = re.subn('[^ACGTN]', 'N', sequence)
            ambiguous_bases += n_ambiguous

            total = None
            total_bases = 0
            step = self.window_size * self.batch_size
            for batch_start in range(0, len(sequence), step):
                chunks = [
                    sequence[start:start + self.window_size]
                    for start in range(batch_start,
                                       min(batch_start + step, len(sequence)),
                                       self.window_size)
                ]
                tokens = self.tokenizer(
                    chunks, return_tensors='pt', padding=True,
                    truncation=False)
                max_positions = getattr(self.model.config,
                                        'max_position_embeddings', None)
                if max_positions is not None and tokens['input_ids'].shape[1] > max_positions:
                    raise ValueError(
                        'A Genos-m window exceeds the model context length; '
                        'reduce --genos-window-size')
                tokens = {key: value.to(self.model.device)
                          for key, value in tokens.items()}
                with torch.inference_mode():
                    states = self.model(**tokens, use_cache=False).last_hidden_state
                    mask = tokens['attention_mask'].unsqueeze(-1)
                    pooled = (states.float() * mask).sum(dim=1) / \
                        mask.sum(dim=1).clamp_min(1)
                    pooled = pooled.cpu().numpy()

                for chunk, vector in zip(chunks, pooled, strict=True):
                    weighted = vector * len(chunk)
                    total = weighted if total is None else total + weighted
                    total_bases += len(chunk)

            vectors.append(total / total_bases)
            if index % 100 == 0 or index == len(contig_ids):
                logger.info('Genos-m embedded %d/%d contigs', index, len(contig_ids))

        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError('Genos-m returned invalid contig embeddings')
        # Translation does not change Euclidean distances or neighbor order.
        # It prevents a large common activation offset from dominating the
        # abundance weighting in SemiBin's later reclustering step.
        embeddings -= embeddings.mean(axis=0, keepdims=True)
        if ambiguous_bases:
            logger.warning('Replaced %d non-ACGTN bases with N for Genos-m',
                           ambiguous_bases)
        logger.info('Genos-m embeddings have shape %s', embeddings.shape)
        return embeddings


@lru_cache(maxsize=1)
def get_genos_m_embedder(model_name, window_size, batch_size, device):
    """Reuse the large frozen model across samples in multi_easy_bin."""
    return GenosMEmbedder(model_name, window_size, batch_size, device)


def calibrate_embedding(embedding, distances, target, logger):
    """Globally rescale vectors and candidate-edge distances together.

    A positive scalar preserves nearest-neighbor ranks while matching the
    absolute distance scale expected by SemiBin's graph or DBSCAN cutoffs.
    """
    positive = distances[np.isfinite(distances) & (distances > 0)]
    if positive.size == 0:
        logger.warning('Genos-m has no positive candidate-edge distances to calibrate')
        return embedding, distances
    q90 = float(np.quantile(positive, 0.9))
    scale = q90 / target
    logger.info('Genos-m distance calibration: q90=%.4g, scale=%.4g',
                q90, scale)
    return embedding / scale, distances / scale
