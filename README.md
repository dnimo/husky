<div><img src="img/Name_Card.png"/></div>

# husky

This project is released by KuoCh"ing Chang.

We are shooting this project to solve medical information processing. 

All rights belong to the author.

# Project map

```text
# Structure
Husky
    __init__.py
    data
    tools
        Tokenizers
            MeCab
            SentencePiece
            tokenizers
        dataformat
            sliding window method
        valuation
            rouge
            bleu
            Information entropy
        analysis
            token distribution
            LDA
            KMeans
            Topic cluster
        cleaner
            deduplicate
            delete \n\n
        Knowldge Graph
            ?
        PCW
            Parallel Context Windows
    Model
        RoBERTa
        Open-calm
        LLama-2
        BERT
    ChatUI
        ?
```

## TODO

### Dataset

* Knowledge graph rebuild
* Deduplicated by sim-hash（Done）

### Token Distribution Analysis

* LDA

### training

* sliding windows (Done)

### valuation

* BLUE (Done)
* ROUGE rewrite by Numpy (Done)

### inference

* Parallel Context Windows(launching)
