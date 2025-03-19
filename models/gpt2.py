import torch
import pytorch_lightning as L
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from utils.config import config


class GPT2(L.LightningModule):
    def __init__(self, tokenizer, learning_rate=0.0001, n_embd=config.model.gpt2["hidden_size"], n_layer=config.model.gpt2["layers"]):
        super().__init__()
        self.save_hyperparameters()

        gpt2_config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=config.data.max_length,
            n_ctx=config.data.max_length,
            n_embd=n_embd,
            n_layer=n_layer,
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

    def forward(self, input_ids, attention_mask,output_attentions=False,output_hidden_states=False, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,output_attentions=output_attentions,output_hidden_states=output_hidden_states
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
