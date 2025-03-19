import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config
import pytorch_lightning as L
import json
import random
from tqdm import tqdm
from utils.tokenizer import gpt2_tokenizer


class Encoder(nn.Module):
    def __init__(
        self,
        n_positions,
        vocab_size,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        compression_ratio,
        pad_token_id=1000,
        bos_token_id=1001,
        cls_token_id=1002,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

        self.compression_ratio = compression_ratio
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_positions, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_head, dim_feedforward, dropout, batch_first=True
            ),
            n_layers,
        )
        if compression_ratio != 1:
            self.downscale = nn.Linear(d_model * 2, int(d_model * 2 // compression_ratio))

    def forward(self, src):
        b, s = src.shape  # [batch_size, sequence_length]
        chunks = src.reshape(b * (s // 2), 2)
        chunks = torch.cat(
            [
                chunks,
                torch.full((chunks.shape[0], 1), self.cls_token_id, device=src.device),
            ],
            dim=1,
        )

        position_ids = torch.arange(s, device=src.device).unsqueeze(0).expand(b, s)
        chunked_position_ids = position_ids.reshape(b * (s // 2), 2)

        positional_embeddings = self.position_embedding(chunked_position_ids)
        positional_embeddings = torch.cat(
            [
                positional_embeddings,
                torch.zeros(
                    (positional_embeddings.shape[0], 1, self.d_model), device=src.device
                ),
            ],
            dim=1,
        )

        embeddings = self.embedding(chunks) + positional_embeddings
        padding_mask = (chunks == self.pad_token_id).float()

        h = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        h = h[:, :-1, :].reshape(b, s // 2, self.d_model * 2)

        if self.compression_ratio != 1:
            h = self.downscale(h)

        padding_mask = padding_mask[:, 0].reshape(b, s // 2)
        return h, padding_mask


class TrunkTransformer(nn.Module):
    def __init__(
        self, d_model, n_head, dim_feedforward, n_layers, dropout, compression_ratio
    ):
        super(TrunkTransformer, self).__init__()
        self.compression_ratio = compression_ratio
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                int(d_model * 2 // compression_ratio),
                n_head,
                dim_feedforward,
                dropout,
                batch_first=True,
            ),
            n_layers,
        )

    def forward(self, src, mask=None, padding_mask=None):
        return self.transformer(src, mask, padding_mask, is_causal=True)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_positions,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        compression_ratio,
        pad_token_id=1000,
        bos_token_id=1001,
    ):
        super(Decoder, self).__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        self.d_model = int(d_model * 2 // compression_ratio)
        self.compression_ratio = compression_ratio
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(n_positions, self.d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.d_model, n_head, dim_feedforward, dropout, batch_first=True
            ),
            n_layers,
        )

    def forward(self, src, tgt):
        b, s = tgt.shape  # [batch_size, sequence_length]
        tgt = tgt.reshape(-1, 2)
        tgt = torch.cat(
            [
                torch.full((tgt.shape[0], 1), self.bos_token_id, device=src.device),
                tgt[:, :-1],
            ],
            dim=1,
        )

        tgt_key_padding_mask = (tgt == self.pad_token_id).float()

        position_ids = torch.arange(s, device=tgt.device).unsqueeze(0).expand(b, s)
        chunked_position_ids = position_ids.reshape(-1, 2)[:, :-1]

        positional_embeddings = self.position_embedding(chunked_position_ids)
        positional_embeddings = torch.cat(
            [
                torch.zeros(
                    (
                        positional_embeddings.shape[0],
                        1,
                        positional_embeddings.shape[-1],
                    ),
                    device=src.device,
                ),
                positional_embeddings,
            ],
            dim=1,
        )

        tgt = self.embedding(tgt) + positional_embeddings
        src = src.reshape(-1, 1, src.shape[-1])
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[1], device=src.device
        )

        return self.transformer(
            tgt,
            src,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )


class Compressor(L.LightningModule):
    def __init__(self, tokenizer):
        super(Compressor, self).__init__()
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.learning_rate = 0.0001

        self.compression_ratio = config.model.compressor["compression_ratio"]
        self.tokenizer = tokenizer

        self.compressor = Encoder(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            **config.model.compressor
        )
        self.trunk = TrunkTransformer(
            **config.model.bigger_transformer, compression_ratio=self.compression_ratio
        )
        self.decompressor = Decoder(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            compression_ratio=self.compression_ratio,
            **config.model.decompressor
        )
        self.linear = nn.Linear(
            int(config.model.decompressor["d_model"] * 2 // self.compression_ratio), vocab_size
        )

    def forward(self, src, tgt):
        h, padding_mask = self.compressor(src)
        mask = nn.Transformer.generate_square_subsequent_mask(
            h.shape[1], device=src.device
        )
        h = self.trunk(h, mask, padding_mask)
        h = self.decompressor(h, tgt)
        return self.linear(h.reshape(src.shape[0], -1, h.shape[-1]))

    def _common_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        input_ids_shifted, labels_shifted = input_ids[:, :-2], labels[:, 2:]
        output = self(input_ids_shifted, labels_shifted)
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            labels_shifted.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def generate(self, text, max_length, temperature):
        self.eval()
        tokenizer = self.tokenizer
        input_ids = tokenizer(text, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(self.device)
        
        if input_ids.shape[1] % 2 != 0:
            return None

        num_tokens_decoded = 0
        while num_tokens_decoded < max_length // 2:
            if input_ids.shape[1] == 2:
                targets = torch.tensor([[self.pad_token_id, self.pad_token_id]]).to(
                    self.device
                )
            else:
                targets = input_ids[:, 2:].clone()
                targets = torch.cat(
                    [
                        targets,
                        torch.tensor([[self.pad_token_id, self.pad_token_id]]).to(
                            self.device
                        ),
                    ],
                    dim=1,
                )

            output = self(input_ids, targets)
            next_token_logits = output[:, -2:, :]
            next_token_logits = next_token_logits / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            first_token = torch.multinomial(
                next_token_probs[0, 0, :], num_samples=1
            ).squeeze(-1)

            if first_token == tokenizer.eos_token_id:
                break

            targets = torch.cat(
                [
                    targets[:, :-2],
                    torch.tensor([[first_token, self.pad_token_id]]).to(self.device),
                ],
                dim=1,
            )

            output = self(input_ids, targets)
            next_token_logits = output[:, -2:, :]
            next_token_logits = next_token_logits / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            second_token = torch.multinomial(
                next_token_probs[0, 1, :], num_samples=1
            ).squeeze(-1)

            if second_token == tokenizer.eos_token_id:
                break

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[first_token, second_token]]).to(self.device),
                ],
                dim=-1,
            )
            num_tokens_decoded += 2

        return tokenizer.decode(input_ids[0])


def main():
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    src = torch.randint(0, 100, (2, 14)).to(device)
    tgt = src.clone()
    tokenizer = gpt2_tokenizer()
    model = Compressor.load_from_checkpoint(
        "checkpoints/concat-epoch=00-val_loss=1.91.ckpt", tokenizer=tokenizer
    ).to(device)
    # model = Compressor(tokenizer).to(device)
    loss = model.validation_step({"input_ids": src, "labels": tgt}, 0)
    print("loss:", loss)



if __name__ == "__main__":
    main()
