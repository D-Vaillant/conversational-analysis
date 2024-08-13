# dataloaders.py
"""This lets you access a Milvus DB and load its contents (according to some filter) into RAM.
    Specifically, we use it here to access a label and an embedding, which we do a neural network over.
"""
import logging
from functools import cached_property
import torch
from torch.utils.data import Dataset, DataLoader
from pymilvus import connections, Collection
from sklearn.model_selection import train_test_split


friends = ["Monica Geller", "Phoebe Buffay", "Joey Tribbiani", "Chandler Bing", "Ross Geller", "Rachel Green"]

class MilvusInMemoryDataset(Dataset):
    def __init__(self, collection_name, fields,
                 label_field='speaker',
                 batch_size=1024, expr=''):
        self.fields = fields
        self.batch_size = batch_size
        self.collection = Collection(collection_name)
        self.collection.load()
        self.speaker_to_label = {speaker: i for i, speaker in enumerate(friends)}
        
        # Create a QueryIterator over collection with the fields + filter.
        self.data = []
        iterator = self.collection.query_iterator(
            expr=expr,
            output_fields=self.fields,
            batch_size=self.batch_size
        )
        i = 0
        ix = 0
        # Iterate over this QueryIterator
        logging.info(f"Querying at batch_size={self.batch_size}")
        while (batch := iterator.next()):
            ix += 1
            i += self.batch_size
            logging.info(f"On Batch#{ix}, x={i}")
            if label_field == 'speaker':
                fy = lambda y: self.speaker_to_label[y]
            elif label_field == 'season':
                fy = lambda y: int(y[1:])-1  # because it has to be zero indexed and seasons are 1-indexed.
            self.data.extend((item['embedding'], fy(item[label_field])) for item in batch)
        
    @cached_property
    def embeddings(self):
        return torch.tensor([item[0] for item in self.data], dtype=torch.float32)
    
    @cached_property
    def labels(self):
        return torch.tensor([item[1] for item in self.data], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
  
        # i = 0
        # while (batch := iterator.next()):
        #     i += self.batch_size
        #     print(f"next batch ({i})")
        #     for item in batch:
        #         if item['speaker'] in self.speaker_to_label:
        #             embedding = torch.tensor(item['embedding'], dtype=torch.float32)
        #             label = torch.tensor(self.speaker_to_label[item['speaker']], dtype=torch.long)
        #             yield embedding, label


def create_dataloaders(collection_name, fields,
                       label_field='speaker', expr='',
                       host='localhost', port=19530,
                       batch_size=32, test_size=0.2, random_state=42):
    """ Establishes the connection to Milvus, """
    # Connect to Milvus via the Milvus ORM.
    connections.connect(host=host, port=port)
    
    # Create dataset
    full_dataset = MilvusInMemoryDataset(collection_name, fields, label_field=label_field,
                                         expr=expr, batch_size=batch_size)
    
    # Perform train/test split
    train_indices, test_indices = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=full_dataset.labels
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader
