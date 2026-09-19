"""KerasLoader — Keras / TensorFlow SavedModel, via conversion to ONNX.

The converted model is a NEW ARTEFACT with its own digest, handed to `ONNXLoader` like any
other ONNX file (plan C6). Its handle records what it was converted from.

STATUS: the conversion path is UNEXERCISED in this build. TensorFlow and tf2onnx are not in
the bundle, and adding TensorFlow would dwarf the <4 GB target on its own. Without them this
loader raises `LoaderUnavailable` naming exactly what is missing, so a Keras artefact yields
an honest "cannot be assessed here" rather than a stack trace or a silent skip. If the two
packages are installed the code below runs, but nothing here has been tested against them.

Refusals that stay regardless of what is installed: legacy HDF5 (`.h5`, `.hdf5`) is refused,
because it can embed a pickled `Lambda` layer that executes on load. SavedModel is converted
from its graph without instantiating Keras layers, and `.keras` is loaded with `safe_mode`.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

from cva.loaders.safety import UnsafeArtifact

from .onnx import OnnxHandle

REQUIRED = ("tensorflow", "tf2onnx")


class LoaderUnavailable(RuntimeError):
    """A loader that recognises the artefact but cannot run here. Distinct from "no loader
    supports this": the format IS supported, the environment is what is missing."""


class KerasLoader:
    name = "KerasLoader"

    def supports(self, path: Path) -> bool:
        path = Path(path)
        if path.is_dir():
            return (path / "saved_model.pb").is_file()
        return path.suffix.lower() in (".keras", ".h5", ".hdf5")

    def load(self, path: Path, model_id: str | None = None) -> OnnxHandle:
        path = Path(path)
        if path.suffix.lower() in (".h5", ".hdf5"):
            raise UnsafeArtifact(path, "S2", "legacy HDF5 can embed a pickled Lambda layer that "
                                 "executes on load; re-save it as .keras or a SavedModel")
        missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
        if missing:
            raise LoaderUnavailable(
                f"{path.name} is a Keras/SavedModel artefact, which is converted to ONNX before "
                f"it is scanned, and {' and '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not installed in this environment. "
                "TensorFlow is not part of the offline bundle. Convert the model to ONNX "
                "elsewhere and scan that file; the converted file is a new artefact with its "
                "own digest.")
        import tf2onnx
        out = Path(tempfile.mkdtemp(prefix="cva-keras-")) / f"{path.stem}.onnx"
        if path.is_dir():
            tf2onnx.convert.from_saved_model(str(path), output_path=str(out))
        else:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(path), safe_mode=True)
            tf2onnx.convert.from_keras(model, output_path=str(out))
        handle = OnnxHandle(out, model_id or path.stem)
        handle.converted_from = str(path)                 # type: ignore[attr-defined]
        return handle
