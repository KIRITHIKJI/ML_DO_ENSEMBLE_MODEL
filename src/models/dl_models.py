"""
STEP 4.2 - The two deep models: BiLSTM and CNN-LSTM.

Both are MULTI-TASK: one network reads the last 48 hours and produces, in
one pass, every output the project needs:

    regression      4 numbers, one per horizon (6, 12, 24, 48 h)
    classification  4 x 3 logits (Low / Medium / Good per horizon)

The regression head predicts the CHANGE in DO, in scaled units, rather than
the DO value itself:

    prediction = DO(t) + head_output * (standard deviation of DO on train)

That is much easier to learn than the absolute value - the network starts
from "no change" instead of having to rediscover the level of every river -
and it is still scored on the final DO value, exactly like every other model.

The station goes in as a small embedding (8 numbers learned per station)
concatenated after the recurrent part, not as a number inside the sequence.
"""

import torch
import torch.nn as nn

from src.config import load_config

CFG = load_config()
DL = CFG["deep_learning"]
HORIZONS = CFG["horizons"]
N_HORIZONS = len(HORIZONS)
N_CLASSES = len(CFG["target"]["class_names"])


class FeatureBranch(nn.Module):
    """
    STEP 5 (b) - the hybrid input.

    The 211 engineered features at time t, squeezed through one dense layer
    and concatenated with the sequence encoder's output. The idea is to give
    the network the lags and rolling statistics the tree models get, instead
    of making it rediscover them from the raw hourly series.
    """

    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_features = hidden

    def forward(self, f):
        return self.net(f)


class Heads(nn.Module):
    """The shared dense layer and the two output heads, used by both models."""

    def __init__(self, in_features, dense_size, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression = nn.Linear(dense_size, N_HORIZONS)
        self.classification = nn.Linear(dense_size, N_HORIZONS * N_CLASSES)

    def forward(self, z):
        z = self.body(z)
        reg = self.regression(z)                       # (B, 4)
        cls = self.classification(z)                   # (B, 12)
        return reg, cls.view(-1, N_HORIZONS, N_CLASSES)


class BiLSTM(nn.Module):
    """
    Two bidirectional LSTM layers over the 48-hour window. The final hidden
    state of the last layer (forward and backward halves) is concatenated
    with the station embedding and sent to the heads.
    """

    name = "bilstm"

    def __init__(self, n_channels, n_stations, cfg=None, n_features=0):
        super().__init__()
        c = dict(CFG["models"]["bilstm"])
        c.update(cfg or {})
        self.embedding = nn.Embedding(n_stations, DL["station_embedding_dim"])
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=c["hidden_size"],
            num_layers=c["num_layers"], bidirectional=c["bidirectional"],
            dropout=c["dropout"] if c["num_layers"] > 1 else 0.0,
            batch_first=True)
        directions = 2 if c["bidirectional"] else 1
        width = c["hidden_size"] * directions + DL["station_embedding_dim"]
        self.features = None
        if n_features:
            self.features = FeatureBranch(n_features, c.get("feature_hidden", 64),
                                          c["dropout"])
            width += self.features.out_features
        self.heads = Heads(width, c["dense_size"], c["dropout"])
        self.directions = directions

    def forward(self, x, station, features=None):
        # h_n is (num_layers * directions, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        if self.directions == 2:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)   # forward + backward
        else:
            last = h_n[-1]
        parts = [last, self.embedding(station)]
        if self.features is not None:
            parts.append(self.features(features))
        return self.heads(torch.cat(parts, dim=1))


class CNNLSTM(nn.Module):
    """
    Two 1-D convolutions pick out short patterns (a sudden drop, a spike),
    max pooling halves the length, and an LSTM reads what is left.
    """

    name = "cnn_lstm"

    def __init__(self, n_channels, n_stations, cfg=None, n_features=0):
        super().__init__()
        c = dict(CFG["models"]["cnn_lstm"])
        c.update(cfg or {})
        ch1, ch2 = c["conv_channels"]
        k = c["kernel_size"]
        self.embedding = nn.Embedding(n_stations, DL["station_embedding_dim"])
        # Conv1d wants (batch, channels, time), so the input is transposed.
        # padding keeps the length at 48 before pooling.
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, ch1, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.Conv1d(ch1, ch2, kernel_size=k, padding=k // 2),
            nn.ReLU(),
            nn.MaxPool1d(c["pool_size"]),
        )
        self.lstm = nn.LSTM(input_size=ch2, hidden_size=c["hidden_size"],
                            num_layers=c["num_layers"], batch_first=True)
        width = c["hidden_size"] + DL["station_embedding_dim"]
        self.features = None
        if n_features:
            self.features = FeatureBranch(n_features, c.get("feature_hidden", 64),
                                          c["dropout"])
            width += self.features.out_features
        self.heads = Heads(width, c["dense_size"], c["dropout"])

    def forward(self, x, station, features=None):
        z = self.conv(x.transpose(1, 2))         # (B, channels, time/2)
        _, (h_n, _) = self.lstm(z.transpose(1, 2))
        parts = [h_n[-1], self.embedding(station)]
        if self.features is not None:
            parts.append(self.features(features))
        return self.heads(torch.cat(parts, dim=1))


MODELS = {"bilstm": BiLSTM, "cnn_lstm": CNNLSTM}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """
    Mean squared error on the scaled DO change, plus weighted cross entropy
    on the class, averaged over the four horizons.

    Both parts are MASKED: a horizon whose target is missing for a row
    contributes nothing, instead of being counted as an error of zero.
    `class_weights` is (n_horizons, n_classes), computed from the TRAINING
    class frequencies so that the rare Low class is not ignored.
    """

    def __init__(self, class_weights, cls_weight=None):
        super().__init__()
        self.register_buffer("class_weights", torch.as_tensor(class_weights,
                                                              dtype=torch.float32))
        self.cls_weight = (DL["classification_loss_weight"]
                           if cls_weight is None else cls_weight)

    def forward(self, reg_out, cls_out, reg_target, cls_target):
        """
        reg_out, reg_target   (B, 4)   scaled DO change; NaN where unknown
        cls_out               (B, 4, 3) logits
        cls_target            (B, 4)   class 0/1/2, or -1 where unknown
        """
        reg_ok = ~torch.isnan(reg_target)
        if reg_ok.any():
            diff = (reg_out - torch.nan_to_num(reg_target)) ** 2
            # mean per horizon over the rows that have a target, then over horizons
            per_h = (diff * reg_ok).sum(0) / reg_ok.sum(0).clamp(min=1)
            reg_loss = per_h[reg_ok.any(0)].mean()
        else:
            reg_loss = reg_out.sum() * 0.0

        cls_losses = []
        for h in range(cls_out.shape[1]):
            ok = cls_target[:, h] >= 0
            if not ok.any():
                continue
            loss_h = nn.functional.cross_entropy(
                cls_out[ok, h, :], cls_target[ok, h].long(),
                weight=self.class_weights[h])
            cls_losses.append(loss_h)
        cls_loss = (torch.stack(cls_losses).mean() if cls_losses
                    else cls_out.sum() * 0.0)

        total = reg_loss + self.cls_weight * cls_loss
        return total, reg_loss.detach(), cls_loss.detach()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
