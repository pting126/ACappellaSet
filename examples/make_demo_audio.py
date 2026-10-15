"""Generate reproducible tones for an installation smoke test, not a quality demo."""
import argparse
from array import array
import math
from pathlib import Path
import tempfile
import wave

RATE=44100


def write_tones(path,seconds,frequencies):
    frames=round(seconds*RATE);samples=array('h')
    for index in range(frames):
        envelope=min(1.,index/441.,(frames-1-index)/441.)
        left=envelope*sum(.12*math.sin(2*math.pi*f*index/RATE) for f in frequencies)
        right=envelope*sum(.12*math.sin(2*math.pi*f*index/RATE+.1) for f in frequencies)
        samples.extend((round(left*32767),round(right*32767)))
    import sys
    if sys.byteorder!='little':samples.byteswap()
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(RATE);stream.writeframes(samples.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():parser.error('Output must be a new directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.demo-',dir=args.output.parent) as temp:
        staging=Path(temp)/'audio';staging.mkdir()
        write_tones(staging/'mixture.wav',1.,[220,330])
        write_tones(staging/'enrollment.wav',10.,[220])
        staging.rename(args.output)
    print(f'Created synthetic tones in {args.output}; no singing or separation quality is demonstrated.')


if __name__=='__main__':main()
