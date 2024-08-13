# conversational-analysis

We analyze data taken from: https://zissou.infosci.cornell.edu/convokit/datasets/friends-corpus/

For the ingestion/Milvus part, uses:
* pymilvus
* llama-index
* llama-index-vector-stores-milvus

For the torch part, uses torch and Pytorch Lightning.

If you're using `poetry`, that should hopefully make installing the requirements a breeze. Hopefully.

## Ingestion
Make sure that you get Milvus running. By default we have a setup that uses a GPU, but you can rename `docker-compose-no-gpu.yml` and use that one instead. Start it up with:
`docker-compose up`

You can download the friends corpus and unzip it, if you wish, but the `utterances.jsonl` file is included.

Then run `ingestion.py`. It connects to the Milvus instances, parses the friends-corpus, and loads it into Milvus. This takes some time. You can play around with the Milvus DB using `milvus_cookbook.ipynb`.

## Machine Learning
We've implemented a couple of simple MLP models using Pytorch, with Lightning to simplify some stuff. You can train them by running `train.py`. They load the data from Milvus via a custom dataloader and run 50 epochs for both the vanilla MLP and the residual MLP. You can specify if the label is the speaker or the season.