# NCCV — a short course on computer vision and image processing

NCCV is an interactive course covering computer vision and image processing
with Python, OpenCV, NumPy, and NVIDIA CUDA. The course starts with webcam
capture and frame processing, then builds toward computational kernels,
convolution, pooling, neural networks, and deeper computer-vision topics.

The lessons are executable [marimo](https://marimo.io) notebooks with reactive
controls. Click on the lesson to launch it in molab, then select the NVIDIA GPU 
from the notebook specs menu

| Lesson | Topics | Launch |
|---|---|---|
| **[1 · Processing webcam frames with OpenCV on GPU](1_processing_webcam_frames_with_opencv_on_gpu.py)** | Browser webcam capture, RGB arrays, mirroring, grayscale, Canny edges, JPEG vs. PNG | [![Open lesson 1 in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ktaletsk/NCCV/blob/master/1_processing_webcam_frames_with_opencv_on_gpu.py/server) |
| **[2 · Writing CUDA kernels for convolution and pooling](2_cuda_kernels_and_convolution.py)** | 2D launch grids, luminance, convolution matrices, max pooling, CPU vs. NVIDIA CUDA | [![Open lesson 2 in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ktaletsk/NCCV/blob/master/2_cuda_kernels_and_convolution.py/server) |

## Course flow

### 1 · Processing webcam frames with OpenCV on GPU

Capture a frame from the front or rear camera, mirror it, and apply color,
grayscale, or Canny edge transforms with OpenCV. Encode the result in memory as
JPEG or PNG and compare size and processing time. The resulting NumPy arrays
are ready for the GPU pipeline in lesson 2.

Further reading: [Displaying real-time webcam streams in Python notebooks](https://medium.com/@kostal91/displaying-real-time-webcam-stream-in-ipython-at-relatively-high-framerate-8e67428ac522)

### 2 · Writing CUDA kernels for convolution and pooling

Follow a webcam frame or generated test card through the complete pipeline:

```mermaid
flowchart TB
    subgraph sources["Frame source"]
        direction LR
        webcam(["Browser webcam<br/>wigglystuff"])
        test_card(["Generated test card"])
    end

    rgb["RGB NumPy frame<br/>uint8 · H × W × 3"]
    path{"Execution path"}

    webcam --> rgb
    test_card --> rgb
    rgb --> path

    subgraph cpu["OpenCV + NumPy reference"]
        direction LR
        cpu_gray["RGB → luminance"]
        cpu_conv["3 × 3 convolution<br/>three edge filters"]
        cpu_pool["5 × 5 max pooling"]
        cpu_gray --> cpu_conv --> cpu_pool
    end

    subgraph gpu["NVIDIA CUDA"]
        direction LR
        upload["Host → device"]
        gpu_gray["Luminance kernel<br/>one thread per pixel"]
        gpu_conv["Convolution kernel<br/>three launches"]
        gpu_pool["Max-pooling kernel"]
        download["Device → host"]
        upload --> gpu_gray --> gpu_conv --> gpu_pool --> download
    end

    path --> cpu_gray
    path --> upload

    combine["Normalize and concatenate"]
    output(["Side-by-side edge responses"])

    cpu_pool --> combine
    download --> combine
    combine --> output
```

The GPU implementation uses NVIDIA's
[`numba-cuda`](https://nvidia.github.io/numba-cuda/) package with CUDA 12. This
project targets NVIDIA CUDA only.

## Run locally

Install [uv](https://docs.astral.sh/uv/), clone the repository, and open either
notebook from the lesson table:

```bash
uvx marimo@0.23.13 edit --sandbox <notebook.py>
```

Marimo creates an isolated environment and installs the notebook's inline
dependencies on first run.

## Stack

- [marimo](https://marimo.io) for reactive notebooks and apps
- [wigglystuff](https://github.com/koaning/wigglystuff) for browser webcam capture
- [OpenCV](https://opencv.org) and NumPy for image processing and CPU references
- [Numba-CUDA](https://nvidia.github.io/numba-cuda/) for NVIDIA GPU kernels
- [uv](https://docs.astral.sh/uv/) for isolated notebook environments

## Roadmap

- **Lesson 1:** webcam capture and OpenCV processing
- **Lesson 2:** CUDA kernels, convolution, and max pooling
- **Lesson 3 (in the works):** connect the operations into a first neural network
- **Later lessons:** deeper computer vision and image-processing topics
