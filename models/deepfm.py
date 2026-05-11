import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import DEVICE, DEFAULT_MODEL_PARAMS


class DeepFM(nn.Module):
    def __init__(self, cat_field_dims, num_input_dim, params=None):
        super().__init__()
        params = params or DEFAULT_MODEL_PARAMS
        self.embed_dim = params["embedding_dim"]
        self.dnn_hidden_units = params["dnn_hidden_units"]
        self.dropout_rate = params["dnn_dropout"]
        self.l2_reg = params["l2_reg"]

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, self.embed_dim) for dim in cat_field_dims
        ])

        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

        self.fm_first_order = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in cat_field_dims
        ])

        self.fm_bias = nn.Parameter(torch.zeros(1))

        total_cat_dim = len(cat_field_dims) * self.embed_dim
        dnn_input_dim = total_cat_dim + num_input_dim
        dnn_layers = []
        prev_dim = dnn_input_dim
        for hu in self.dnn_hidden_units:
            dnn_layers.append(nn.Linear(prev_dim, hu))
            dnn_layers.append(nn.BatchNorm1d(hu))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hu
        dnn_layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.dnn:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, cat_inputs, num_input):
        cat_inputs = cat_inputs.long()
        embed_list = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]

        fm_first = torch.stack([self.fm_first_order[i](cat_inputs[:, i]) for i in range(len(self.fm_first_order))], dim=1)
        fm_first = fm_first.sum(dim=1)

        fm_second = torch.stack(embed_list, dim=1)
        sum_square = fm_second.sum(dim=1).pow(2)
        square_sum = fm_second.pow(2).sum(dim=1)
        fm_second = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)

        dnn_cat_input = torch.cat(embed_list, dim=1)
        dnn_input = torch.cat([dnn_cat_input, num_input], dim=1)
        dnn_out = self.dnn(dnn_input)

        output = self.fm_bias + fm_first.sum(dim=1, keepdim=True) + fm_second + dnn_out
        output = torch.sigmoid(output)
        return output.squeeze(-1)

    def get_regularization_loss(self):
        reg_loss = 0.0
        for emb in self.embeddings:
            reg_loss += torch.sum(emb.weight ** 2)
        for linear in self.dnn:
            if isinstance(linear, nn.Linear):
                reg_loss += torch.sum(linear.weight ** 2)
        return self.l2_reg * reg_loss


class DeepFMTrainer:
    def __init__(self, model, params=None):
        self.model = model.to(DEVICE)
        self.params = params or DEFAULT_MODEL_PARAMS
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.params["learning_rate"],
            weight_decay=self.params["l2_reg"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.criterion = nn.BCELoss()
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            cat_inputs, num_input, labels = [b.to(DEVICE) for b in batch]
            self.optimizer.zero_grad()
            outputs = self.model(cat_inputs, num_input)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for batch in dataloader:
            cat_inputs, num_input, labels = [b.to(DEVICE) for b in batch]
            outputs = self.model(cat_inputs, num_input)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        auc = compute_auc(all_labels, all_preds)
        return avg_loss, auc, np.array(all_preds), np.array(all_labels)

    def fit(self, train_loader, val_loader, epochs=None, verbose=True):
        epochs = epochs or self.params["epochs"]
        patience = self.params.get("early_stopping_patience", 5)
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc, _, _ = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            self.scheduler.step(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return self.history


def compute_auc(labels, preds):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, preds)
    except ValueError:
        return 0.5


class RecommendDataset(torch.utils.data.Dataset):
    def __init__(self, X_cat, X_num_user, X_num_prod, y, text_embeddings=None):
        self.X_cat = torch.LongTensor(X_cat)
        if text_embeddings is not None:
            self.X_num = torch.FloatTensor(np.concatenate([X_num_user, X_num_prod, text_embeddings], axis=1))
        else:
            self.X_num = torch.FloatTensor(np.concatenate([X_num_user, X_num_prod], axis=1))
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]
