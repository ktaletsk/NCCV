# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cuda-toolkit[cccl,cudart,nvcc,nvrtc]==13.0.*; sys_platform == 'linux'",
#     "marimo==0.23.13",
#     "numba-cuda-mlir[cu13]==0.4.1; sys_platform == 'linux'",
#     "nvidia-nvjitlink==13.0.*; sys_platform == 'linux'",
#     "numpy>=2.2,<2.4",
#     "opencv-python-headless==4.13.0.92",
#     "pillow>=11,<13",
#     "wigglystuff[pillow]==0.5.10",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


with app.setup(hide_code=True):
    from time import perf_counter

    import cv2
    import marimo as mo
    import numpy as np
    from wigglystuff import WebcamCapture

    try:
        from numba_cuda_mlir import cuda
    except (ImportError, ModuleNotFoundError):
        cuda = None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lesson 2. Writing CUDA kernels for convolution and pooling

    A **computational kernel** is a small function run independently across
    many data elements. For an image, a two-dimensional CUDA grid lets one GPU
    thread own one pixel. We will convert RGB to luminance, convolve the image
    with three edge-detection kernels, and max-pool the results.

    Notice that we are not writing C++ CUDA here, but rather using the `numba-cuda-mlir` 
    compiler.

    In this exercise, we will first implement different kernels in OpenCV/NumPy to 
    run on the CPU. And after that, we will use Numba to write raw CUDA kernels and run them 
    using Numba JIT compiler.
    """)
    return


@app.cell
def _():
    cuda_available = False
    cuda_detail = "numba-cuda-mlir is not installed in this environment"
    if cuda is not None:
        try:
            cuda_available = cuda.is_available()
            if cuda_available:
                device_name = cuda.current_context().device.name
                if isinstance(device_name, bytes):
                    device_name = device_name.decode()
                cuda_detail = str(device_name)
            else:
                cuda_detail = (
                    "numba-cuda-mlir is installed, but no CUDA device is attached"
                )
        except Exception as exc:
            cuda_detail = f"CUDA initialization failed: {exc}"

    mo.callout(
        f"CUDA ready: **{cuda_detail}**"
        if cuda_available
        else f"CPU reference mode: **{cuda_detail}**",
        kind="success" if cuda_available else "info",
    )
    return cuda_available, cuda_detail


@app.cell
def _():
    capture_interval = mo.ui.slider(
        250,
        2_000,
        step=250,
        value=1_000,
        label="Auto-capture interval (ms)",
        show_value=True,
    )
    facing_mode = mo.ui.dropdown(
        {"Front camera": "user", "Rear camera": "environment"},
        value="Front camera",
        label="Camera",
    )
    mo.hstack([capture_interval, facing_mode], justify="start", gap=1)
    return capture_interval, facing_mode


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
                "Use a webcam frame, or continue without camera permission to "
                "process the generated test card."
            ),
            webcam,
        ]
    )
    return (webcam,)


@app.cell
def _():
    def get_frame(camera) -> np.ndarray | None:
        """Read the latest browser capture as a mirrored RGB NumPy array."""
        pil_frame = camera.get_pil()
        if pil_frame is None:
            return None
        rgb_frame = np.asarray(pil_frame.convert("RGB"))
        return cv2.flip(rgb_frame, 1)

    return (get_frame,)


@app.cell
def _():
    def make_test_frame(height: int = 360, width: int = 480) -> np.ndarray:
        """Build a deterministic RGB test card with edges in many directions."""
        y, x = np.mgrid[:height, :width]
        checker = (((x // 40) + (y // 40)) % 2) * 90
        frame = np.stack(
            [
                np.clip(40 + x * 215 / width, 0, 255),
                np.clip(40 + y * 215 / height, 0, 255),
                np.clip(70 + checker, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 75, (250, 245, 80), -1)
        cv2.rectangle(frame, (35, 35), (150, 130), (40, 210, 240), 8)
        cv2.line(frame, (20, height - 30), (width - 20, 25), (245, 60, 80), 9)
        return frame

    return (make_test_frame,)


@app.cell
def _(get_frame, make_test_frame, webcam):
    captured_frame = get_frame(webcam)
    if captured_frame is None:
        frame = make_test_frame()
        source_label = "generated test card"
    else:
        frame = captured_frame
        source_label = "browser webcam"

    mo.vstack(
        [
            mo.md(f"### Input: {source_label}"),
            mo.image(frame, alt=source_label, width="100%", rounded=True),
        ]
    )
    return frame, source_label


@app.cell
def _():
    if cuda is None:
        blur_kernel = None
    else:

        @cuda.jit
        def blur_kernel(x, out):
            blur_size = 3
            i, j = cuda.grid(2)

            if i < x.shape[0] and j < x.shape[1]:
                for channel in range(3):
                    out[i, j, channel] = 0
                count = 0
                for ii in range(-blur_size, blur_size + 1):
                    for jj in range(-blur_size, blur_size + 1):
                        row = i + ii
                        col = j + jj
                        if 0 <= row < x.shape[0] and 0 <= col < x.shape[1]:
                            for channel in range(3):
                                out[i, j, channel] += x[row, col, channel]
                            count += 1
                for channel in range(3):
                    out[i, j, channel] /= count

    return (blur_kernel,)


@app.cell
def _():
    if cuda is None:
        bw_kernel = None
    else:

        @cuda.jit
        def bw_kernel(x, out):
            i, j = cuda.grid(2)
            if i < x.shape[0] and j < x.shape[1]:
                out[i, j] = (
                    0.2126 * x[i, j, 0]
                    + 0.7152 * x[i, j, 1]
                    + 0.0722 * x[i, j, 2]
                )

    return (bw_kernel,)


@app.cell
def _():
    if cuda is None:
        conv_kernel = None
    else:

        @cuda.jit
        def conv_kernel(x, kernel, out):
            kernel_size = (kernel.shape[0] - 1) // 2
            i, j = cuda.grid(2)

            if i < x.shape[0] and j < x.shape[1]:
                out[i, j] = 0
                for ii in range(-kernel_size, kernel_size + 1):
                    for jj in range(-kernel_size, kernel_size + 1):
                        row = i + ii
                        col = j + jj
                        if 0 <= row < x.shape[0] and 0 <= col < x.shape[1]:
                            out[i, j] += (
                                x[row, col]
                                * kernel[ii + kernel_size, jj + kernel_size]
                            )

    return (conv_kernel,)


@app.cell
def _():
    if cuda is None:
        max_pool = None
    else:

        @cuda.jit
        def max_pool(x, out):
            i, j = cuda.grid(2)

            if i < out.shape[0] and j < out.shape[1]:
                max_brightness = 0.0
                for ii in range(5):
                    for jj in range(5):
                        value = x[5 * i + ii, 5 * j + jj]
                        if value > max_brightness:
                            max_brightness = value
                out[i, j] = max_brightness

    return (max_pool,)


@app.cell
def _():
    # Try different convolution kernels.
    gauss_blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    edge_detector_1 = np.array(
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32
    )
    edge_detector_2 = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32
    )
    edge_detector_3 = np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
    )
    sharpen = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    )
    convolution_kernels = (
        edge_detector_1,
        edge_detector_2,
        edge_detector_3,
    )
    mo.md(
        f"""
        ### The filters

        The filter bank contains Gaussian blur, three edge detectors, and a
        sharpening kernel. The processing pipeline applies the three edge
        detectors side by side.

        ```text
        Edge 1                 Edge 2                 Edge 3
        {edge_detector_1}     {edge_detector_2}     {edge_detector_3}
        ```
        """
    )
    return convolution_kernels, gauss_blur, sharpen


@app.cell
def _():
    def max_pool_cpu(image: np.ndarray, pool_size: int = 5) -> np.ndarray:
        """Reference implementation of non-overlapping max pooling."""
        height = image.shape[0] // pool_size
        width = image.shape[1] // pool_size
        cropped = image[: height * pool_size, : width * pool_size]
        windows = cropped.reshape(height, pool_size, width, pool_size)
        return windows.max(axis=(1, 3))

    return (max_pool_cpu,)


@app.cell
def _(max_pool_cpu):
    def cpu_pipeline(
        rgb_frame: np.ndarray, kernels: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, float]:
        """OpenCV/NumPy reference for the CUDA pipeline."""
        started = perf_counter()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        convolved = [
            cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
            for kernel in kernels
        ]
        # The CUDA kernel initializes each pooling window at zero, so negative
        # edge responses are clipped here too for a like-for-like reference.
        pooled = [max_pool_cpu(np.maximum(response, 0.0)) for response in convolved]
        normalized = [
            cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for response in pooled
        ]
        result = np.concatenate(normalized, axis=1)
        return result, perf_counter() - started

    return (cpu_pipeline,)


@app.cell
def _(convolution_kernels, cpu_pipeline, frame):
    cpu_result, cpu_elapsed = cpu_pipeline(frame, convolution_kernels)
    mo.vstack(
        [
            mo.md("### CPU reference"),
            mo.image(
                cpu_result,
                alt="Three convolved and max-pooled CPU outputs",
                width="100%",
                rounded=True,
            ),
            mo.md(
                f"OpenCV + NumPy completed the pipeline in "
                f"**{cpu_elapsed * 1_000:.1f} ms**."
            ),
        ]
    )
    return


@app.cell
def _():
    def run_cuda_pipeline(
        rgb_frame: np.ndarray,
        kernels: tuple[np.ndarray, ...],
        grayscale_kernel,
        convolution_kernel,
        pooling_kernel,
    ) -> tuple[np.ndarray, float]:
        """Transfer one frame to CUDA, run the kernels, and copy results back."""
        started = perf_counter()
        source = np.ascontiguousarray(rgb_frame, dtype=np.float32)
        height, width = source.shape[:2]
        pooled_shape = (height // 5, width // 5)
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (height + threads_per_block[0] - 1) // threads_per_block[0],
            (width + threads_per_block[1] - 1) // threads_per_block[1],
        )
        pooled_blocks = (
            (pooled_shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (pooled_shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
        )

        device_frame = cuda.to_device(source)
        device_gray = cuda.device_array((height, width), dtype=np.float32)
        grayscale_kernel[blocks_per_grid, threads_per_block](
            device_frame, device_gray
        )

        host_outputs = []
        for kernel in kernels:
            device_kernel = cuda.to_device(kernel)
            device_convolved = cuda.device_array((height, width), dtype=np.float32)
            device_pooled = cuda.device_array(pooled_shape, dtype=np.float32)
            convolution_kernel[blocks_per_grid, threads_per_block](
                device_gray, device_kernel, device_convolved
            )
            pooling_kernel[pooled_blocks, threads_per_block](
                device_convolved, device_pooled
            )
            host_outputs.append(device_pooled.copy_to_host())

        cuda.synchronize()
        normalized = [
            cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for response in host_outputs
        ]
        result = np.concatenate(normalized, axis=1)
        return result, perf_counter() - started

    return (run_cuda_pipeline,)


@app.cell
def _(cuda_available):
    run_gpu = mo.ui.run_button(
        label="Run CUDA pipeline", disabled=not cuda_available
    )
    run_gpu
    return (run_gpu,)


@app.cell
def _(
    bw_kernel,
    conv_kernel,
    convolution_kernels,
    cuda_available,
    frame,
    max_pool,
    run_cuda_pipeline,
    run_gpu,
):
    mo.stop(
        not cuda_available,
        mo.callout(
            "Attach an NVIDIA GPU (in molab, use the notebook specs "
            "button) to enable this comparison.",
            kind="info",
        ),
    )
    mo.stop(
        not run_gpu.value,
        mo.callout(
            "CUDA is ready. Click **Run CUDA pipeline** to compile the "
            "kernels and process the current frame.",
            kind="neutral",
        ),
    )

    try:
        gpu_result, gpu_elapsed = run_cuda_pipeline(
            frame,
            convolution_kernels,
            bw_kernel,
            conv_kernel,
            max_pool,
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("### CUDA result"),
                    mo.image(
                        gpu_result,
                        alt="Three convolved and max-pooled CUDA outputs",
                        width="100%",
                        rounded=True,
                    ),
                    mo.md(
                        f"Transfer + first/next kernel execution completed in "
                        f"**{gpu_elapsed * 1_000:.1f} ms**. The first click "
                        "also includes JIT compilation."
                    ),
                ]
            )
        )
    except Exception as exc:
        mo.output.replace(
            mo.callout(f"CUDA execution failed: `{exc}`", kind="danger")
        )
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(
            """
            The launch grid is derived from the frame dimensions so every pixel
            receives a thread. Pooling uses non-overlapping `5 × 5` windows,
            and all five convolution matrices remain available for experiments.
            """
        ),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
