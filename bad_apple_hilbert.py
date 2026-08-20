import argparse
import subprocess
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Render Bad Apple Hilbert lines + right-side partitioned 1D stream bar + synced audio!"
    )
    parser.add_argument("-p", "--order", type=int, default=6, help="Hilbert order p")
    parser.add_argument("-v", "--video", type=str, default="bad_apple.mp4", help="Path to input video")
    parser.add_argument("-o", "--output", type=str, default="bad_apple_hilbert.mp4", help="Output MP4 filename")
    parser.add_argument("-s", "--scale", type=int, default=8, help="Scale factor per Hilbert node")
    parser.add_argument("--strip-thickness", type=int, default=64, help="Width of right 1D stream bar in px")
    parser.add_argument("--divider-thickness", type=int, default=4, help="Width of vertical partition line in px")
    parser.add_argument("--divider-color", type=int, default=128, help="Grayscale value for partition (0-255)")
    parser.add_argument("-r", "--fps", type=int, default=30, help="Framerate")
    args = parser.parse_args()

    p = args.order
    N = 2**p
    num_points = N * N
    scale = args.scale
    canvas_dim = N * scale
    out_w = canvas_dim + args.divider_thickness + args.strip_thickness
    out_h = canvas_dim

    # 1. Precompute Hilbert Coordinates & Line Slices
    hc = HilbertCurve(p, 2)
    distances = np.arange(num_points)
    pts = np.array(hc.points_from_distances(distances))
    y_coords, x_coords = pts[:, 1], pts[:, 0]

    # Precompute pixel centers on the high-res canvas
    py = y_coords * scale + scale // 2
    px = x_coords * scale + scale // 2

    # Precompute Manhattan line segment bounds
    p1_y, p2_y = py[:-1], py[1:]
    p1_x, p2_x = px[:-1], px[1:]

    y_min, y_max = np.minimum(p1_y, p2_y), np.maximum(p1_y, p2_y) + 1
    x_min, x_max = np.minimum(p1_x, p2_x), np.maximum(p1_x, p2_x) + 1

    th = max(1, scale // 4)

    # 1D stream bar mapping (sampled across canvas_dim height)
    idx_1d = np.linspace(0, num_points - 1, canvas_dim).astype(np.int64)

    # Pre-generate vertical partition divider
    divider = np.full(
        (canvas_dim, args.divider_thickness),
        args.divider_color,
        dtype=np.uint8,
    )

    # 2. FFmpeg input pipe (N x N grayscale)
    ffmpeg_in = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", args.video,
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{N}x{N}", "-"
        ],
        stdout=subprocess.PIPE,
        bufsize=num_points,
    )

    # 3. FFmpeg output pipe (Raw video stdin + Original audio direct copy)
    ffmpeg_out = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{out_w}x{out_h}",
            "-r", str(args.fps),
            "-i", "-",               # Input 0: Raw video pipe from Python stdin
            "-i", args.video,        # Input 1: Original video file for audio track
            "-map", "0:v:0",         # Take video from input 0 (Hilbert render)
            "-map", "1:a:0?",        # Take audio from input 1 (if audio stream exists)
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "veryfast",
            "-c:a", "copy",          # Direct audio copy without re-encoding
            "-shortest",             # Finish cleanly when either stream ends
            args.output,
        ],
        stdin=subprocess.PIPE,
    )

    print(f"Rendering lines, right partitioned stream bar, and audio to {args.output}...")

    try:
        while True:
            raw_frame = ffmpeg_in.stdout.read(num_points)
            if len(raw_frame) < num_points:
                break

            # 1D Hilbert serialization
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((N, N))
            hilbert_1d = frame[y_coords, x_coords]

            # A. Render Hilbert lines onto empty canvas
            canvas = np.zeros((canvas_dim, canvas_dim), dtype=np.uint8)
            vals = hilbert_1d[:-1]
            for y1, y2, x1, x2, val in zip(y_min, y_max, x_min, x_max, vals):
                canvas[max(0, y1 - th):min(canvas_dim, y2 + th),
                       max(0, x1 - th):min(canvas_dim, x2 + th)] = val

            # B. Tile right-side stream bar horizontally
            sampled_1d = hilbert_1d[idx_1d]
            strip = np.tile(sampled_1d.reshape(canvas_dim, 1), (1, args.strip_thickness))

            # C. Composite canvas + vertical divider + stream bar horizontally
            composite = np.hstack([canvas, divider, strip])
            ffmpeg_out.stdin.write(composite.tobytes())

    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        ffmpeg_in.stdout.close()
        ffmpeg_in.wait()
        if ffmpeg_out.stdin:
            ffmpeg_out.stdin.close()
        ffmpeg_out.wait()

    print(f"Finished rendering to {args.output}.")


if __name__ == "__main__":
    main()
