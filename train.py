import os
import logging

import dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from model import LightningMLPClassifier
from dataloaders import create_dataloaders
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="mlp_classifier")

dotenv.load_dotenv()
USE_TENSOR_CORES = os.getenv('USE_TENSOR_CORES', False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_TENSOR_CORES:
    try:
        assert torch.cuda.is_available()
        torch.set_float32_matmul_precision('medium' | 'high')
    except AssertionError:
        # Silently ignore.
        USE_TENSOR_CORES = False
        logging.warning("USE_TENSOR_CORES enabled despite CUDA not being installed properly.")


# Usage
collection_name = "openai_02"
fields = ['embedding', 'speaker']
# fields = ['embedding', 'season']
y = fields[1]

friends = ["Monica Geller", "Phoebe Buffay", "Joey Tribbiani", "Chandler Bing", "Ross Geller", "Rachel Green"]
expr = 'speaker in [{}]'.format(', '.join(['"{}"'.format(f) for f in friends]))
train_loader, test_loader = create_dataloaders(collection_name, fields, label_field=y, expr=expr, batch_size=1024)

input_dim = 1536
hidden_dim = 256
output_dim = 6 if y == 'speaker' else 10

model = LightningMLPClassifier(input_dim, hidden_dim, output_dim)
model.to(device)

trainer = L.Trainer(max_epochs=25, accelerator='auto',
                    log_every_n_steps=1,
                    logger=logger)
trainer.fit(model, train_loader, test_loader)