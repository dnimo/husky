<div align="center"><img src="img/Name_Card.png"/></div>

# husky

This project is release by KuoCh"ing Chang.

We shooting this project to solve medical info. processing. 

All rights belong to Auther.

# Project map

```text
# Structure
Husky
    tools
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
* deduplicate by sim-hash（Done）

### Token Distribution Analysis

* RoBerta (lanuching)
* LDA
* Information entropy

### training

* sliding windows (Done)

### valuation

* BLUE
* ROUGE rewrite by Numpy

### inference

* Parallel Context Windows
