from transformers import BertModel, BertConfig
from torch import nn


class BertSummary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits
