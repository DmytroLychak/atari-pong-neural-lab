"""Portable SB3 checkpoints, including the Mac's original Python 3.12.0.

Some early Python zip streams cannot be used directly by the nested PyTorch
ZIP reader. Keep the SB3 archive, but store its tensor state dictionaries in
PyTorch's supported legacy stream format. Tensor values and PPO metadata are
unchanged; PPO.load remains the normal public loading API.
"""

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import torch


def convert_archive(source, destination):
    source, destination = Path(source), Path(destination)
    with ZipFile(source) as archive, ZipFile(destination, "w", ZIP_DEFLATED) as output:
        for name in archive.namelist():
            data = archive.read(name)
            if name.endswith(".pth"):
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                stream = io.BytesIO()
                torch.save(state, stream, _use_new_zipfile_serialization=False)
                data = stream.getvalue()
            output.writestr(name, data)


def save_model(model, path):
    path = Path(path)
    if path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pong-save-", dir=path.parent) as temporary:
        raw = Path(temporary) / "raw.zip"
        ready = Path(temporary) / "ready.zip"
        model.save(raw)
        convert_archive(raw, ready)
        os.replace(ready, path)
