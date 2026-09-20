"""Print the zip directory of a TorchScript archive and of a torch.save checkpoint.

Item 22: `supports()` must tell the two apart WITHOUT deserialising. This records what the
two layouts actually look like on this torch version, so the sniff is written against
observed bytes rather than against memory of the format.
"""
import io
import zipfile

import torch
import torch.nn as nn

ts = io.BytesIO()
torch.jit.save(torch.jit.script(nn.Linear(4, 3)), ts)
print("TORCHSCRIPT:", sorted(zipfile.ZipFile(io.BytesIO(ts.getvalue())).namelist())[:12])

sv = io.BytesIO()
torch.save({"state_dict": nn.Linear(4, 3).state_dict(), "arch": "x"}, sv)
print("TORCHSAVE  :", sorted(zipfile.ZipFile(io.BytesIO(sv.getvalue())).namelist())[:12])
