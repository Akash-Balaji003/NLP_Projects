import lightning as L
import torch
import torch.nn.functional as F
import wandb
from create_dataloader import CreateDataLoaders
from dataset import my_load_dataset
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmotionClassifier(L.LightningModule):
    """Logistic regression on pooled word embeddings (pooling: mean | sum | lstm)."""

    def __init__(
        self,
        weights,
        n_classes,
        lr=1e-3,
        pooling="mean",
        freeze_embedding=True,
        lstm_hidden=128,
        init_std=0.02,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weights"])  # logged to W&B automatically

        self.emb = nn.Embedding.from_pretrained(
            weights, freeze=freeze_embedding, padding_idx=0
        )
        dim = weights.shape[1]

        if pooling == "lstm":
            self.lstm = nn.LSTM(dim, lstm_hidden, batch_first=True)
            dim = lstm_hidden
        self.fc = nn.Linear(dim, n_classes)

        self._init_weights(init_std)

    def _init_weights(self, std):
        # Gaussian init for the new layers (the pretrained embedding stays untouched)
        nn.init.normal_(self.fc.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc.bias)
        if self.hparams.pooling == "lstm":
            for name, p in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.normal_(p, mean=0.0, std=std)
                else:
                    nn.init.zeros_(p)

    def forward(self, ids):  # ids: (batch, seq_len)
        mask = ids != 0  # True for real tokens
        x = self.emb(ids)  # (batch, seq_len, dim)
        lengths = mask.sum(1)

        if self.hparams.pooling == "sum":
            h = x.sum(dim=1)  # padding rows are zero
        elif self.hparams.pooling == "mean":
            h = x.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)
        elif self.hparams.pooling == "lstm":
            packed = pack_padded_sequence(
                x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
            h = hn[-1]  # last hidden state, (batch, hidden)
        else:
            raise ValueError(f"unknown pooling: {self.hparams.pooling}")

        return self.fc(h)  # logits

    def _step(self, batch, stage):
        ids, y = batch  # y: one-hot float (batch, n_classes)
        logits = self(ids)
        loss = F.cross_entropy(logits, y)  # accepts probability targets
        acc = (logits.argmax(1) == y.argmax(1)).float().mean()
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=ids.size(0),
        )
        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=ids.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


def train_model(
    train_loader,
    val_loader,
    weights,
    n_classes,
    logger=None,
    max_epochs=30,
    **model_kwargs,
):
    model = EmotionClassifier(weights, n_classes, **model_kwargs)
    if logger is not None:
        # histograms of the classification layer's weights and bias (+ gradients)
        logger.watch(model.fc, log="all", log_freq=100)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[EarlyStopping("val_loss", patience=5), LearningRateMonitor("epoch")],
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


def start_wandb_sweep():
    dataset, emotions = my_load_dataset()
    train_loader, val_loader, test_loader, weights = CreateDataLoaders(
        dataset, emotions
    ).create_dataloaders()
    n_classes = len(emotions)

    # --- sweep over the initial learning rate ---
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
            "pooling": {
                "values": ["mean", "sum"]
            },  # optional, drop if only lr is wanted
        },
    }

    def sweep_run():
        with wandb.init() as run:
            logger = WandbLogger(experiment=run)
            cfg = run.config
            train_model(
                train_loader,
                val_loader,
                weights,
                n_classes,
                logger=logger,
                lr=cfg.lr,
                pooling=cfg.pooling,
            )

    sweep_id = wandb.sweep(sweep_config, project="goemotions")
    wandb.agent(sweep_id, function=sweep_run, count=20)


def start_sinlge_run():
    dataset, emotions = my_load_dataset()
    train_loader, val_loader, test_loader, weights = CreateDataLoaders(
        dataset, emotions
    ).create_dataloaders()
    n_classes = len(emotions)

    # --- single run ---
    logger = WandbLogger(project="goemotions", name="mean-pool-baseline")
    model, trainer = train_model(
        train_loader,
        val_loader,
        weights,
        n_classes,
        logger=logger,
        lr=1e-3,
        pooling="mean",
    )
    trainer.test(model, test_loader)
    wandb.finish()


if __name__ == "__main__":
    start_sinlge_run()
