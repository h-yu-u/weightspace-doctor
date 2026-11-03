import pytest
import torch
from torch import nn

# Keep tiny-model CPU tests bounded on machines with many logical cores.
torch.set_num_threads(1)


class TinyTokenizer:
    chat_template = None

    def __call__(self, text, *, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[ord(c) % 127 for c in text]], dtype=torch.long)}


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(5, 4, bias=False)

    def forward(self, x):
        h = torch.cat([x, x[..., :1]], dim=-1)
        return self.down_proj(torch.tanh(h))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, use_cache=False):
        x = self.emb(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return TinyModel(), TinyTokenizer()
