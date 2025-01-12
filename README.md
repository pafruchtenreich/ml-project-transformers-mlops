# Transformer Exploration for News Summarization :newspaper:

This project explores Transformer-based models inspired by the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). We compare three approaches for **news article summarization** using the [Kaggle News Summarization dataset](https://www.kaggle.com/datasets/sbhatti/news-summarization):

1. :repeat: A **traditional RNN encoder-decoder** model  
2. :zap: A **Transformer** model (as in *Attention Is All You Need*)  
3. :rocket: A **pre-trained BERT** model (fine-tuned for summarization)

---

## Results and Experiments

| Model                        | ROUGE-1 | ROUGE-2 | ROUGE-L | Train Time (ep)  | Params  |
|------------------------------|---------|---------|---------|------------------|---------|
| RNN Encoder-Decoder          | ...     | ...     | ...     | ~ $7.7 \times 10^4$ s (25) | $1.19 \times 10^7$  |
| Transformer                  | ...     | ...     | ...     | ...              | $1.25 \times 10^7$  |
| BERT                         | ...     | ...     | ...     | ~ $1.1 \times 10^5$ s (10) | $1.33 \times 10^8$  |


