import os
import logging

import dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from model import LightningMLPClassifier, LightningResidualMLP
from dataloaders import create_dataloaders
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

check_friends=True
if check_friends:
    postfix = '[friends]'
else:
    postfix = '[seasons]'
logger = TensorBoardLogger("tb_logs", name=f"mlp_classifier{postfix}")
logger2 = TensorBoardLogger("tb_logs", name=f"residual_classifier{postfix}")

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
if check_friends:
    fields = ['embedding', 'speaker']
else:
    fields = ['embedding', 'season']
y = fields[1]

friends = ["Monica Geller", "Phoebe Buffay", "Joey Tribbiani", "Chandler Bing", "Ross Geller", "Rachel Green"]
expr = 'speaker in [{}]'.format(', '.join(['"{}"'.format(f) for f in friends]))
train_loader, test_loader = create_dataloaders(collection_name, fields, label_field=y, expr=expr, batch_size=128)

input_dim = 1536
hidden_dim = 256
output_dim = 6 if y == 'speaker' else 10

mlp = LightningMLPClassifier(input_dim, hidden_dim, output_dim)
res = LightningResidualMLP(input_dim, hidden_dim, output_dim)

for m, l in zip([mlp, res], [logger, logger2]):
    m.to(device)
    trainer = L.Trainer(max_epochs=50, accelerator='auto',
                        log_every_n_steps=5,
                        #callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=15)],
                        logger=l)
    trainer.fit(m, train_loader, test_loader)