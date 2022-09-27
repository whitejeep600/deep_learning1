from typing import Dict

import torch
from torch.nn import Embedding, RNN, Linear, Softmax


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = RNN(input_size=embeddings.size(dim=1), hidden_size=hidden_size, num_layers=num_layers,
                       dropout=dropout, bidirectional=bidirectional)
        self.num_class = num_class
        dimension_multiplier = 2 if bidirectional else 1
        self.final_linear = Linear(hidden_size * dimension_multiplier, num_class)

    @property
    def encoder_output_size(self) -> int:
        return self.num_class

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        output, hidden_state = self.rnn(batch)
        output_for_last_tokens = output[-1, :, :]  # taking only the output for the last token
        s = Softmax(dim=2)
        probabilities = s(self.final_linear(output_for_last_tokens))
        return {'prediction': probabilities}
        # todo co z tym typem zwracanym


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
