from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlm_pretraining import BertEncoder, ModelConfig


@dataclass
class DecoderConfig:
    """Configuration class for the custom Transformer decoder."""
    vocab_size: int
    hidden_size: int = 512
    num_layers: int = 4
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 256
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)


class DecoderEmbeddings(nn.Module):
    """Constructs the token and position embeddings for the decoder."""
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_embeddings(input_ids) + self.position_embeddings(pos)
        x = self.layer_norm(x)
        return self.dropout(x)


class HybridDecoderBlock(nn.Module):
    """
    Middle-path decoder block:
    - Uses low-level nn.MultiheadAttention + Linear + LayerNorm
    - Avoids nn.TransformerDecoder / nn.TransformerDecoderLayer
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.ffn_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.ffn_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.norm3 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm masked self-attention.
        x_ln = self.norm1(x)
        self_out, _ = self.self_attn(
            query=x_ln,
            key=x_ln,
            value=x_ln,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(self_out)

        # Pre-norm cross-attention (decoder -> encoder memory).
        x_ln = self.norm2(x)
        cross_out, _ = self.cross_attn(
            query=x_ln,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(cross_out)

        # Pre-norm FFN.
        x_ln = self.norm3(x)
        ffn = self.ffn_fc2(F.gelu(self.ffn_fc1(x_ln)))
        x = x + self.dropout(ffn)
        return x


class HybridDecoderStack(nn.Module):
    """A stack of custom HybridDecoderBlocks."""
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.layers = nn.ModuleList([HybridDecoderBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x


class GenerativeQAModelHybrid(nn.Module):
    """Main generative QA model using the custom hybrid decoder stack."""

    def __init__(self, encoder_cfg: ModelConfig, decoder_cfg: DecoderConfig):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.encoder = BertEncoder(encoder_cfg)

        if encoder_cfg.hidden_size != decoder_cfg.hidden_size:
            self.enc_to_dec = nn.Linear(encoder_cfg.hidden_size, decoder_cfg.hidden_size)
        else:
            self.enc_to_dec = nn.Identity()

        self.decoder_embeddings = DecoderEmbeddings(decoder_cfg)
        self.decoder = HybridDecoderStack(decoder_cfg)
        self.final_ln = nn.LayerNorm(decoder_cfg.hidden_size, eps=decoder_cfg.layer_norm_eps)
        self.lm_head = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.vocab_size, bias=False)
        self.lm_head.weight = self.decoder_embeddings.token_embeddings.weight

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Generates a causal (upper triangular) mask to prevent attention to future tokens."""
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Passes the input through the BERT encoder and projects it to the decoder's hidden size."""
        enc = self.encoder(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        return self.enc_to_dec(enc)

    def decode(
        self,
        memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the decoder stack using the encoder's memory, applying causal masking to target tokens."""
        dec_inp = self.decoder_embeddings(decoder_input_ids)
        tgt_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
        tgt_key_padding_mask = decoder_attention_mask == 0
        memory_key_padding_mask = encoder_attention_mask == 0

        dec_out = self.decoder(
            x=dec_inp,
            memory=memory,
            tgt_attn_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        dec_out = self.final_ln(dec_out)
        return self.lm_head(dec_out)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> dict:
        """Performs a full forward pass (encoding + decoding) and computes cross-entropy loss if labels are provided."""
        memory = self.encode(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        logits = self.decode(memory, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
        out = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def sequence_logprob(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        pad_token_id: int,
        normalize_by_length: bool = True,
    ) -> torch.Tensor:
        """Computes the log probability of a specific target sequence given the encoder inputs.
        Used for evaluating the likelihood of the 'No Answer' sequence during inference gating."""
        if target_ids.size(1) < 2:
            return torch.zeros(target_ids.size(0), device=target_ids.device)

        memory = self.encode(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        dec_in = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        dec_mask = (dec_in != pad_token_id).long()
        logits = self.decode(memory, encoder_attention_mask, dec_in, dec_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        valid = labels != pad_token_id
        seq_logp = (token_logp * valid).sum(dim=1)
        if not normalize_by_length:
            return seq_logp
        lengths = valid.sum(dim=1).clamp(min=1)
        return seq_logp / lengths

    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_token_type_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int = 32,
        beam_size: int = 4,
        length_penalty: float = 1.0,
        return_logprob: bool = False,
    ):
        """Generates sequences using beam search decoding."""
        assert encoder_input_ids.size(0) == 1, "generate currently supports batch size 1."
        memory = self.encode(encoder_input_ids, encoder_token_type_ids, encoder_attention_mask)
        beams = [(torch.tensor([[bos_token_id]], device=memory.device), 0.0, False)]

        for _ in range(max_new_tokens):
            candidates = []
            for seq, score, ended in beams:
                if ended:
                    candidates.append((seq, score, True))
                    continue
                dec_attn = torch.ones_like(seq, device=seq.device)
                logits = self.decode(memory, encoder_attention_mask, seq, dec_attn)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                # Expand each sequence in the beam with the top-k most likely next tokens
                topk = torch.topk(log_probs, k=beam_size, dim=-1)
                for i in range(beam_size):
                    tok = topk.indices[0, i].item()
                    tok_logp = topk.values[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[tok]], device=seq.device)], dim=1)
                    is_end = tok == eos_token_id
                    candidates.append((new_seq, score + tok_logp, is_end))

            def norm_score(item):
                # Apply length penalty to the score to avoid heavily penalizing longer sequences
                seq, score, _ = item
                denom = max(1.0, float(seq.size(1)) ** length_penalty)
                return score / denom

            candidates.sort(key=norm_score, reverse=True)
            beams = candidates[:beam_size]
            if all(x[2] for x in beams):
                break

        best_seq, best_score, _ = beams[0]
        best_len = best_seq.size(1)
        best = best_seq
        if best_len < max_new_tokens + 1:
            pad_len = max_new_tokens + 1 - best_len
            best = torch.cat(
                [best, torch.full((1, pad_len), pad_token_id, device=best.device, dtype=best.dtype)], dim=1
            )
        if return_logprob:
            token_count = max(1, best_len - 1)
            return best, float(best_score) / float(token_count), best_len
        return best

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder_top_layers(self, n_layers: int) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        n_layers = max(0, min(n_layers, len(self.encoder.layers)))
        for layer in self.encoder.layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.encoder.final_ln.parameters():
            p.requires_grad = True

    def unfreeze_encoder_all(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

GenerativeQAModel = GenerativeQAModelHybrid
