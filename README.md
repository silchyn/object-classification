# object-classification
![demonstration](https://user-images.githubusercontent.com/30828805/168285649-943d10c7-b382-434c-ba9b-379cfda775ca.png)

A project that detects and classifies objects in an image using the [YOLO](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) algorithm with OpenCV.

For the program to work, you need to additionally download [yolo.weigths](https://drive.google.com/file/d/1hBtwdYTSkdSfU6mcoWWvN2Czrye2e3lx/view?usp=sharing). To run the program on your own photo, replace the `target.jpg` file with the photo, keeping the file name, and run `main.py`.  
Before running the program, check that the file hierarchy is exactly like this:  
```markup
└── object-classification
    ├── main.py
    ├── target.jpg
    ├── yolo.cfg
    ├── yolo.names
    └── yolo.weights
```
