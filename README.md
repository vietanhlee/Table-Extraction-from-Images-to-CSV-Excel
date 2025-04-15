# Description

This project is about extracting structured table data from images or PDF files into CSV/Excel formats. This is especially useful for automating the extraction of financial reports or other structured documents. The project uses
PaddleOCR for text detection to identify word-level bounding boxes. These detected boxes are then passed to EasyOCR's recognition backend, allowing for parallel OCR processing, which significantly improves both speed and accuracy
compared to processing each box individually. A post-processing module is applied to group recognized texts into appropriate table columns and rows and convert them into structured CSV or Excel output.

>Note that: this model is just a compiling model, which means that I have simply gathered scripts from models in order to create a cohesive and comprehensive result. The end-to-end project will be started in the near future.

# Outline

1. Text Detection (`Paddleocr`)
2. Text Recognition (use function recognizer of `Easyocr`'s backend)
3. Convert data into structured CSV or Excel format

# Text Dectection

Text detection is the process of locating text in an image or video and recognizing the presence of characters. The [DB algorithm](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md) is a popular algorithm used in the [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/quick_start.html) framework to localize text in the input image. It works by detecting the text regions in the image and then grouping them into text lines. This algorithm is known for its high accuracy and speed.

<!-- To enhance the accuracy of Text Recognition, images cropped by the DB algorithm were padded. This is because the padding helps to ensure that the text is not cut off during the recognition process. -->

# Text Recognition

Text Recognition is the process of recognizing the text in an image or video. For Text Recognition part, i used [Easyocr](https://github.com/JaidedAI/EasyOCR), which is a popular framework for OCR task. In Vietnamese ocr task, i feel it is the best ocr library at recognizing. 

# Usage

Firstly, clone this repository by executing in CMD:

```
git clone https://github.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel
```

After cloning the repository, download the required dependencies by running:
- For using cpu:

    ```
    pip install -r requirements_cpu.txt
    ```

- For using gpu:

    ```
    pip install -r requirements_gpu.txt
    ```

For command-line usage, execute the following script for inference:

```
python predict.py --img path_image
```

> example: 
    `python predict.py --image test\7.jpg`

For Jupyter Notebook, you can explore and experiment with the code at [predict.ipynb](https://github.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel/predict.ipynb).

# References

- [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/quick_start.html)
- [Easyocr](https://github.com/JaidedAI/EasyOCR)