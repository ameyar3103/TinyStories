import torch
import pytorch_lightning as L
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from utils.config import config
import random

def shuffle_groups_between_ones(tensor, one_id=1, ten_id=10):
    result = tensor.clone()
    for row_idx in range(tensor.shape[0]):
        row = tensor[row_idx]
        ten_idx = (row == ten_id).nonzero(as_tuple=True)[0][0]
        one_indices = (row[:ten_idx] == one_id).nonzero(as_tuple=True)[0]
        if len(one_indices) < 2:
            continue
        groups = []
        start_idx = 0
        for one_idx in one_indices:
            group = row[start_idx:one_idx+1].tolist()
            groups.append(group)
            start_idx = one_idx + 1
        if start_idx < ten_idx:
            groups.append(row[start_idx:ten_idx].tolist())
        random.shuffle(groups)
        new_sequence = []
        for group in groups:
            new_sequence.extend(group)
        result[row_idx, :ten_idx] = torch.tensor(new_sequence)
    return result

class GPT2(L.LightningModule):
    def __init__(self, tokenizer, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()

        gpt2_config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=config.data.max_length,
            n_ctx=config.data.max_length,
            n_embd=config.model.gpt2["hidden_size"],
            n_layer=config.model.gpt2["layers"],
            n_head=config.model.gpt2["heads"],
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.model = GPT2LMHeadModel(gpt2_config)
        self.config = gpt2_config

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.learning_rate = learning_rate

        self.val_predictions = []
        self.val_targets = []

        self.test_generations = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        full_stop_idx = tokenizer.encode(".")[0]

        print(input_ids.shape, full_stop_idx, tokenizer.eos_token_id)
         
        contrastive_inputs = shuffle_groups_between_ones(input_ids,full_stop_idx, tokenizer.eos_token_id)

        contrastive_inputs = contrastive_inputs[:contrastive_inputs.shape[0]//3, :].to(input_ids.device)

        outputs = self(
            input_ids=contrastive_inputs,
            attention_mask=attention_mask,
            labels=contrastive_inputs,
        )

        con_loss = outputs.loss

        loss += 0.3/(con_loss + 0.01)


        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        self.log("val_loss", loss)

        return loss

    def on_validation_epoch_end(self):
        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=config.data.max_length,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_return_sequences=1,
        )

        decoded_outputs = [
            self.tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True) for output in output_ids
        ]
        self.test_generations.extend(decoded_outputs)

        return decoded_outputs

    def on_test_epoch_end(self):
        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    from utils.tokenizer import gpt2_tokenizer

    tokenizer = gpt2_tokenizer()
    model = GPT2(tokenizer)

    # sample batch
    input_ids, labels = torch.randint(0, len(tokenizer), (4, 128)), torch.randint(
        0, len(tokenizer), (4, 128)
    )

    input_ids[:, -1] = tokenizer.eos_token_id
    attention_mask = torch.ones_like(input_ids)

    loss = model.training_step(
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}, 0
    )

    print(loss)

    val_loss = model.validation_step(
        {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}, 0
    )

    print(val_loss)

    test_generations = model.test_step({"input_ids": input_ids}, 0)

    print(test_generations)
