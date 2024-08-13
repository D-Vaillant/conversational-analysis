import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_blocks=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

        
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=4, l2_lambda=1e-4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # L2 regularization
        self.l2_lambda = l2_lambda
        
    def forward(self, x):
        x = self.input_bn(x)
        for layer in self.layers:
            x = layer(x)
        return x


class LightningResidualMLP(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3, learning_rate=1e-3, l2_lambda=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResidualMLP(input_dim, hidden_dim, output_dim, num_blocks)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        l2_reg = sum(param.pow(2.0).sum() for param in self.parameters())
        loss += self.hparams.l2_lambda * l2_reg
        # Calculate accuracy.
        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # Calculate accuracy.
        _, predicted = torch.max(logits, 1)
        acc = (predicted == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.criterion(logits, y)
    #     self.log('test_loss', loss)
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        

class LightningMLPClassifier(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = MLPClassifier(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters())