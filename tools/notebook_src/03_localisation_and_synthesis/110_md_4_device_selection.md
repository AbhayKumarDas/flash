## 4. Device selection

Times the noise generation on CPU and on GPU and uses whichever is faster.

Only the noise field benefits from the GPU. Object detection and compositing are OpenCV
operations, and `cv2.seamlessClone` in particular is CPU only and is the most expensive step in
this notebook. Do not expect the accelerator to change the total runtime much.

The sanity figure below shows one host per category with the detected object region and the
placement blob drawn on it. Check that the blob sits on the product.
