import torch.nn as nn


class BoWModel(nn.Module):
    def __init__(self, embed_dim, embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        """
        emb = self.embedding(input_ids)      # (B, L, D)
        h = emb.mean(dim=-2)                   # (B, D)
        logits = self.linear(h)               # (B, 1)
        return self.sigmoid(logits).squeeze(-1)  # (B)


if __name__ == "__main__":
    from p70 import emb
    import torch
    from torchinfo import summary
    model = BoWModel(emb.size()[1], emb)
    batch_size = 32
    max_length = 128
    summary(model, input_size=[batch_size, max_length], col_names=["output_size", "num_params"], dtypes=[torch.int])
