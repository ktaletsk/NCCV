# Short course on computer vision and image processing

We will be looking at different ways to process images using Python in Jupyter Notebooks.

**Lesson 1.** We start with acquiring images from webcam using OpenCV and efficiently displaying them inside Jupyter Notebook using Pillow library.

🔗 Reference:
https://medium.com/@kostal91/displaying-real-time-webcam-stream-in-ipython-at-relatively-high-framerate-8e67428ac522

📓 Notebook:

🛠 Dependencies installation:
```
conda install jupyter
pip install opencv-python
```

**Lesson 2.** We introduce GPU kernels and CUDA (using Numba) to achieve fast image processing. We introduce convolution operation and convolution kernels to achieve blurring or edge detection. We learn how to apply them using GPU kernels. Finally, we introduce the max pooling layer

🔗 Reference: *Upcoming*

📓 Notebook:

🛠 Dependencies installation:
```
conda install numba
conda install cudatoolkit
```

**Lesson 3 (in the works).** We connect the layers to build our first neural network.

Next lessons are TBD and will dive deeper in different aspects of image processing and computer vision

This course uses Python 3 and requires CUDA capable device (NVIDIA GPU) to run kernels (which may change in the future if I rewrite kernels to use AMD cards and/or CPUs).