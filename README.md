<div align="center"><img src="img/Name_Card.png"/></div>

# husky

This project is released by KuoCh"ing Chang.

We are shooting this project to solve medical info. processing. 

All rights belong to the author.

# Project map

```text
# Structure
Husky
    __init__.py
    data
    tools
        tikenizers
            MeCab
            tokenizers
            __init__.py
        dataset
            sliding window method
        valuation
            rouge
            bleu
        analysis
            token distribution
            LDA
            KMeans
            Topic cluster
        cleaner
            deduplicate
            format
        Knowldge Graph
            ?
    Model
        base
            RoBERTa
            Open-calm
            LLama-2
        train
            lora
        inference
            Parallel Context Window method

```

## TODO

### Dataset

* Knowledge graph rebuild
* Deduplicated by sim-hash（Done）

### Token Distribution Analysis

* RoBerta (lanuching)
* LDA
* Information entropy
* Rreq (lanuching)

### training

* sliding windows (Done)

### valuation

* BLUE (Done)
* ROUGE rewrite by Numpy (Done)

### inference

* Parallel Context Windows
