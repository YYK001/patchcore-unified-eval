"""Fresh-process numerical test helper. Never used in production."""
import sys
from pathlib import Path
import numpy as np
import torch
from external_baselines.patchcore_official_eval.adapter import make_model, ChunkedFlatL2, load_bank, StreamingPatchCore, ChunkedApproximateGreedy


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = torch.nn.Conv2d(3, 8, 3, stride=8, padding=1)
        self.layer3 = torch.nn.Conv2d(8, 12, 3, stride=2, padding=1)

    def forward(self, x):
        return self.layer3(self.layer2(x))


def main():
    path = Path(sys.argv[1])
    def forbidden(*args, **kwargs):
        raise AssertionError('Fresh prediction must not fit/extract training features/select coreset')
    StreamingPatchCore.fit = forbidden
    StreamingPatchCore._fill_memory_bank = forbidden
    ChunkedApproximateGreedy.select = forbidden
    backbone = TinyBackbone()
    backbone.load_state_dict(torch.load(path/'tiny.pth', weights_only=True))
    model = make_model(None, 'cpu', ChunkedFlatL2('cpu', query_chunk=11), backbone=backbone)
    load_bank(model, path/'bank.npy')
    scores, maps = model.predict(torch.from_numpy(np.load(path/'images.npy')))
    np.savez(path/'reloaded.npz', scores=scores, maps=maps)


if __name__ == '__main__':
    main()
