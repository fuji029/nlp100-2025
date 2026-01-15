from torch.nn.utils.rnn import pad_sequence
from p73 import SST2Dataset
from torch.utils.data import DataLoader
import torch


def collate_fn(batch):
    input_ids, labels = list(zip(*batch))

    # pad_sequence の padding_value はデフォルトで `0`
    # pad_sequence の padding_side はデフォルトで `'right'`
    # patch_first=True で B * L にする
    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = torch.stack(labels)
    return input_ids, labels


if __name__ == "__main__":
    from torch import tensor
    sample = [{'text': 'hide new secretions from the parental units',
               'label': tensor(0.),
               'input_ids': tensor([5785,     66, 113845,     18,     12,  15095,   1594])},
              {'text': 'contains no wit , only labored gags',
               'label': tensor(0.),
               'input_ids': tensor([3475,    87, 15888,    90, 27695, 42637])},
              {'text': 'that loves its characters and communicates something rather beautiful about human nature',
               'label': tensor(1.),
               'input_ids': tensor([4,  5053,    45,  3305, 31647,   348,   904,  2815,    47,  1276,  1964])},
              {'text': 'remains utterly satisfied to remain the same throughout',
               'label': tensor(0.),
               'input_ids': tensor([987, 14528,  4941,   873,    12,   208,   898])}]

    sample_data = SST2Dataset([item["input_ids"] for item in sample], [item["label"] for item in sample])
    sample_data = DataLoader(sample_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    for input_ids, labels in sample_data:
        print(input_ids)  # (B, L)
        print(labels)  # (B)
