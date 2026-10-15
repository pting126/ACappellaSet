"""Separate one audio file using a registered ACappellaSet checkpoint."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import tempfile
import time


ROOT = Path(__file__).resolve().parent


def read_audio(path, sample_rate):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    original_channels = audio.shape[1]
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Input audio must be nonempty and finite.")
    if original_channels == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif original_channels != 2:
        raise ValueError("Only mono or stereo input is supported.")
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor, axis=0)
    return np.ascontiguousarray(audio.T), {
        "original_sample_rate": original_rate,
        "original_channels": original_channels,
        "mono_duplicated": original_channels == 1,
        "resampled": original_rate != sample_rate,
        "resampling": "scipy.signal.resample_poly, default Kaiser window",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="combined-ep333", help="Registered model ID")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local checkpoint; SHA-256 is checked")
    parser.add_argument("--input", type=Path, required=True, help="Input audio file")
    parser.add_argument("--output", type=Path, required=True, help="New output directory (must not exist)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU thread count (default: 2)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    registry = json.loads((ROOT / "manifests/models.json").read_text())
    specs = {entry["id"]: entry for entry in registry["models"]}
    if args.model not in specs:
        parser.error(f"Unknown model {args.model!r}. Available: {', '.join(specs)}")
    spec = specs[args.model]
    import torch
    import soundfile as sf
    from models.fixed import load_model, separate, sha256

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    torch.manual_seed(1337)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    started = time.monotonic()
    config_path = ROOT / spec["config"]
    model, config = load_model(args.checkpoint, config_path, spec, device)
    mixture, input_details = read_audio(args.input, spec["sample_rate"])
    estimates = separate(model, config, mixture, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete directory only after all stems and the receipt exist.
    with tempfile.TemporaryDirectory(prefix=".acappellaset-", dir=args.output.parent) as scratch:
        staging = Path(scratch) / "result"
        staging.mkdir()
        outputs = []
        for stem, estimate in zip(spec["source_order"], estimates):
            path = staging / f"{stem}.wav"
            sf.write(path, estimate.T, spec["sample_rate"], subtype="FLOAT")
            outputs.append({"stem": stem, "file": path.name, "sha256": sha256(path)})
        receipt = {
            "model": args.model, "checkpoint_sha256": sha256(args.checkpoint),
            "original_checkpoint_sha256": spec["checkpoint_sha256"],
            "config_sha256": spec["config_sha256"], "input_sha256": sha256(args.input),
            "input": input_details, "source_order": spec["source_order"],
            "sample_rate": spec["sample_rate"], "channels": 2, "frames": mixture.shape[-1],
            "output_subtype": "FLOAT", "gain": 1.0, "outputs": outputs,
            "device": str(device), "threads": args.threads, "seed": 1337,
            "inference": dict(spec["inference"], amp_used=bool(config.training.use_amp)),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in
                         ("torch", "torchaudio", "demucs", "numpy", "scipy", "soundfile", "omegaconf")},
            "elapsed_seconds": time.monotonic() - started,
        }
        (staging / "inference.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        if args.output.exists():
            raise FileExistsError("Output directory appeared during inference; refusing to replace it.")
        staging.rename(args.output)
    print(f"Saved {len(outputs)} stems and inference.json to {args.output}")


if __name__ == "__main__":
    main()
