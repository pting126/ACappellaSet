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
    parser.add_argument("--model", default="Fixed_6stem_htdemucs_SKnJC", help="Registered model ID")
    parser.add_argument("--list-models", action="store_true", help="List supported models without loading weights")
    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint; SHA-256 is checked")
    parser.add_argument("--input", type=Path, help="Input audio file")
    parser.add_argument("--output", type=Path, help="New output directory (must not exist)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU thread count (default: 2)")
    parser.add_argument("--selection", help="Validation criterion; required when multiple selections are released")
    parser.add_argument("--role", help="Part or dual query: S/A/T/B/VP/Lead/Bar")
    parser.add_argument("--enrollment", type=Path, help="Recording of the desired singer")
    parser.add_argument("--enrollment-start", type=float, default=0.0, help="Start time in enrollment recording, seconds")
    parser.add_argument("--allow-untrained-role", action="store_true",
                        help="Explicitly reproduce the historical zero-role diagnostic")
    args = parser.parse_args(argv)
    from models.registry import print_models, resolve, select_checkpoint
    if args.list_models:
        print_models()
        return
    if any(value is None for value in (args.checkpoint, args.input, args.output)):
        parser.error("--checkpoint, --input, and --output are required for inference")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")
    try:
        spec = select_checkpoint(resolve(args.model), args.selection)
        conditioned = spec['family'] == 'conditioned-htdemucs'
        if conditioned:
            from models.query import validate_query
            query = validate_query(spec, args.role, args.enrollment, args.enrollment_start,
                                   args.allow_untrained_role)
        elif args.role is not None or args.enrollment is not None or args.enrollment_start != 0 or args.allow_untrained_role:
            raise ValueError('Fixed and specialist models do not accept conditioned queries')
    except ValueError as exc:
        parser.error(str(exc))
    args.model = spec["id"]
    import torch
    import soundfile as sf
    from models.fixed import sha256
    if conditioned:
        from models.conditioned import load_model, separate
    else:
        from models.fixed import load_model, separate

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
    query_details = None
    if conditioned:
        enrollment = None
        query_details = dict(query, selection=spec['selection'], step=spec['step'])
        if args.enrollment is not None:
            enrollment, enrollment_details = read_enrollment(
                args.enrollment, spec['sample_rate'], args.enrollment_start, config['enroll_seconds'])
            query_details.update(enrollment=enrollment_details, enrollment_sha256=sha256(args.enrollment))
        estimates = separate(model, config, mixture, device, enrollment, query['role_index'])
        amp_used = False
    else:
        estimates = separate(model, config, mixture, device)
        amp_used = bool(config.training.use_amp)
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
            "family": spec["family"], "target_role": spec.get("target_role"),
            "output_contract": spec.get("output_contract"),
            "original_checkpoint_sha256": spec["checkpoint_sha256"],
            "config_sha256": spec["config_sha256"], "input_sha256": sha256(args.input),
            "input": input_details, "source_order": spec["source_order"],
            "sample_rate": spec["sample_rate"], "channels": 2, "frames": mixture.shape[-1],
            "output_subtype": "FLOAT", "gain": 1.0, "outputs": outputs,
            "device": str(device), "threads": args.threads, "seed": 1337,
            "inference": dict(spec["inference"], amp_used=amp_used),
            "query": query_details,
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


def read_enrollment(path, sample_rate, start, seconds):
    """Crop before resampling; preserve the native zero-padding/peak-scale contract."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not math.isfinite(start) or start < 0:
        raise ValueError('Enrollment start must be finite and nonnegative')
    info = sf.info(path)
    offset = round(start * info.samplerate)
    count = round(seconds * info.samplerate)
    if offset >= info.frames:
        raise ValueError('Enrollment start is beyond the available audio')
    audio, rate = sf.read(path, start=offset, frames=count, dtype='float32', always_2d=True)
    if audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
        raise ValueError('Enrollment must contain finite mono or stereo audio')
    actual = len(audio)
    audio = np.pad(audio, ((0, count - actual), (0, 0)))
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor, axis=0)
    wanted = round(seconds * sample_rate)
    audio = audio[:wanted]
    if len(audio) < wanted:
        audio = np.pad(audio, ((0, wanted - len(audio)), (0, 0)))
    return np.ascontiguousarray(audio.T), {
        'start_seconds': start, 'duration_seconds': seconds, 'original_sample_rate': rate,
        'original_channels': info.channels, 'source_frames_read': actual,
        'zero_padded_frames_at_original_rate': count - actual,
        'normalization': 'peak after crop and resampling, as applied by conditioned adapter',
    }


if __name__ == "__main__":
    main()
