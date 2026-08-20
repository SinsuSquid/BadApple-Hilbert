# [BadApple-Hilbert](https://youtu.be/Bp2uJbzMsUc)

[Bad Apple!!](https://www.youtube.com/watch?v=FtutLA63Cp8) but it's on a 1D continuous bitstream mapped onto a 2D Hilbert space-filling curve.

A high-throughput video processing pipeline that decomposes video frames into a continuous 1D Hilbert space-filling curve and reconstructs them into an animated line snapshot video alongside a real-time 1D stream monitor.

The pipeline uses raw inter-process communication (IPC) pipes via `FFmpeg` and vectorized `NumPy` array operations to eliminate disk I/O bottlenecks and third-party computer vision dependencies.

## Features
- True Curve Reconstruction: Rebuilds each frame from scratch by drawing connected Manhattan line segments ordered along the Hilbert path.

- 1D Hilbert Stream Monitor: Displays the serialized 1D pixel sequence alongside the reconstructed video separated by a customizable partition.

- Direct FFmpeg IPC Pipes: Decodes and encodes raw byte streams entirely in-memory using standard POSIX pipes.

- Synchronized Audio Passthrough: Direct-stream copies (`-c:a copy`) the original audio track without transcoding or loss of sync.

- Zero OpenCV / PIL Dependencies: Line rasterization and spatial transforms are computed entirely with NumPy coordinate slicing.

## How It Works
```
[Input Video]
      │ (ffmpeg pipe)
      ▼
[Raw Grayscale Frame (N x N)]
      │
      ├─► [1D Hilbert Curve Sampling] ──► [1D Stream Strip Generation]
      │                                                │
      └─► [Continuous Vector Line Rasterization]       │
                        │                              │
                        ▼                              ▼
                 [Canvas Frame] ─── [Partition] ─── [Stream Strip]
                                       │
                                       ▼ (Horizontal Stack)
                              [Composite Frame]
                                       │ (ffmpeg pipe + original audio)
                                       ▼
                              [Rendered Output MP4]
```
1. **Space-Filling Indexing**: An $N \times N$ Hilbert grid ($N = 2^p$) is precomputed into $(x, y)$ coordinate mappings.
2. **1D Curve Serialization**: Grayscale video frames are serialized into an array of size $N^2$ based on continuous distance along the curve.
3. **Canvas Rasterization**: Connected line segments between consecutive coordinates are drawn onto a blank high-resolution canvas with brightness mapped to the sampled pixel value.
4. **Multiplexing & Output**: The canvas, divider partition, and downsampled 1D stream bar are stacked horizontally and piped directly into an FFmpeg `libx264` encoder with the source audio track multiplexed via `-map`.

## Requirements
- Python 3.8+
- `FFmpeg` installed and accessible via system `PATH`
- Python dependencies:
    - `numpy`
    - `hilbertcurve`

## Usage
### Basic Execution
Run the script against any input video (default: `bad_apple.mp4`):
```bash
python hilbert_render.py -v input.mp4 -o output.mp4
```

