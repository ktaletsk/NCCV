# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.13",
#     "numpy>=2.2,<2.4",
#     "opencv-python-headless==4.13.0.92",
#     "pillow>=11,<13",
#     "wigglystuff[pillow]==0.5.10",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


with app.setup(hide_code=True):
    from io import BytesIO
    from time import perf_counter

    import cv2
    import marimo as mo
    import numpy as np
    from PIL import Image
    from wigglystuff import WebcamCapture


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lesson 1. Processing webcam frames with OpenCV on GPU

    Capture frames in the browser with wigglystuff's `WebcamCapture`, convert
    them to NumPy arrays, and prepare them with OpenCV for the CUDA kernels in
    lesson 2. The reactive pipeline processes each captured frame without a
    blocking video loop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Capture, transform, and encode

    `WebcamCapture` keeps the live preview in the browser and publishes sampled
    frames to Python. Each captured image is converted to an RGB NumPy array,
    mirrored with `cv2.flip`, and passed through an OpenCV transform.

    The processed array is encoded in memory with Pillow. JPEG is lossy and
    usually smaller and faster to transmit; PNG is lossless and better suited
    to sharp masks, diagrams, and synthetic images. Use the controls below to
    compare both formats on the same frame.
    """)
    return


@app.cell
def _():
    capture_interval = mo.ui.slider(
        100,
        2_000,
        step=100,
        value=500,
        label="Auto-capture interval (ms)",
        show_value=True,
    )
    facing_mode = mo.ui.dropdown(
        {"Front camera": "user", "Rear camera": "environment"},
        value="Front camera",
        label="Camera",
    )
    effect = mo.ui.dropdown(
        ["Color", "Grayscale", "Canny edges"],
        value="Color",
        label="OpenCV transform",
    )
    image_format = mo.ui.dropdown(
        ["JPEG", "PNG"], value="JPEG", label="Python encoding"
    )
    mo.hstack(
        [capture_interval, facing_mode, effect, image_format],
        justify="start",
        wrap=True,
        gap=1,
    )
    return capture_interval, effect, facing_mode, image_format


@app.cell
def _(capture_interval, facing_mode):
    webcam = mo.ui.anywidget(
        WebcamCapture(
            interval_ms=capture_interval.value,
            facing_mode=facing_mode.value,
        )
    )
    mo.vstack(
        [
            mo.md(
                "Allow camera access, then use **Capture** once or enable "
                "**Auto capture** for reactive Python processing."
            ),
            webcam,
        ]
    )
    return (webcam,)


@app.cell
def _():
    # Encode frames without writing temporary image files.
    def array_to_image(array: np.ndarray, fmt: str = "jpeg") -> bytes:
        """Encode a NumPy image in memory and return its compressed bytes."""
        stream = BytesIO()
        pil_image = Image.fromarray(array)
        save_format = "JPEG" if fmt.lower() in {"jpeg", "jpg"} else "PNG"
        save_options = {"quality": 88} if save_format == "JPEG" else {}
        pil_image.save(stream, save_format, **save_options)
        return stream.getvalue()

    return (array_to_image,)


@app.cell
def _():
    def get_frame(camera) -> np.ndarray | None:
        """Read the latest browser capture as a mirrored RGB NumPy array."""
        pil_frame = camera.get_pil()
        if pil_frame is None:
            return None

        # Flip the image for natural, mirror-like viewing.
        rgb_frame = np.asarray(pil_frame.convert("RGB"))
        return cv2.flip(rgb_frame, 1)

    return (get_frame,)


@app.cell
def _(array_to_image, effect, get_frame, image_format, webcam):
    frame = get_frame(webcam)
    mo.stop(
        frame is None,
        mo.callout(
            "No frame has reached Python yet. Capture a frame above to "
            "inspect the OpenCV pipeline.",
            kind="info",
        ),
    )

    started = perf_counter()
    if effect.value == "Grayscale":
        processed = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif effect.value == "Canny edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed = cv2.Canny(gray, threshold1=80, threshold2=160)
    else:
        processed = frame

    encoded = array_to_image(processed, image_format.value)
    elapsed = perf_counter() - started
    channels = 1 if processed.ndim == 2 else processed.shape[2]

    mo.vstack(
        [
            mo.image(
                encoded,
                alt=f"Webcam frame after {effect.value.lower()}",
                rounded=True,
            ),
            mo.md(
                f"""
                | Frame | Value |
                |---|---:|
                | Shape | `{processed.shape}` |
                | Channels | {channels} |
                | Encoded as | {image_format.value} |
                | Encoded size | {len(encoded) / 1_024:.1f} KiB |
                | Transform + encode | {elapsed * 1_000:.1f} ms |
                | Single-frame throughput | {1 / elapsed:.1f} FPS |
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(
            """
            **Frame pipeline:** browser webcam → captured PNG → Pillow image →
            RGB NumPy array → OpenCV transform → JPEG or PNG output. Lesson 2
            sends the prepared array through custom NVIDIA CUDA kernels.
            """
        ),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
