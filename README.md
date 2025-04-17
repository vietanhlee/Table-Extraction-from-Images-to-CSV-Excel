# Description

This project is about extracting structured table data from images or PDF files into CSV/Excel formats. This is especially useful for automating the extraction of financial reports or other structured documents. The project uses PaddleOCR for text detection to identify word-level bounding boxes. These detected boxes are then passed to EasyOCR's recognition backend, allowing for parallel OCR processing, which significantly improves both speed and accuracy compared to processing each box individually. A post-processing module is applied to group recognized texts into appropriate table columns and rows and convert them into structured CSV or Excel output.

# Outline

1. Text Detection (`Paddleocr`)
2. Text Recognition (use function recognizer of `Easyocr`'s backend)
3. Convert data into structured CSV or Excel format

# Text Dectection

Text detection is the process of locating text in an image or video and recognizing the presence of characters. The [DB algorithm](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_det_db_en.md) is a popular algorithm used in the [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/quick_start.html) framework to localize text in the input image. It works by detecting the text regions in the image and then grouping them into text lines. This algorithm is known for its high accuracy and speed.

<!-- To enhance the accuracy of Text Recognition, images cropped by the DB algorithm were padded. This is because the padding helps to ensure that the text is not cut off during the recognition process. -->

# Text Recognition

Text Recognition is the process of recognizing the text in an image or video. For Text Recognition part, i used [Easyocr](https://github.com/JaidedAI/EasyOCR), which is a popular framework for OCR task. In Vietnamese ocr task, i find it's the best OCR library at recognizing. In this part, i used PaddleOCR for text detection to identify word-level bounding boxes. These detected boxes are then passed to EasyOCR's recognition backend, allowing for parallel OCR processing, which significantly improves both speed and accuracy compared to processing each box individually 

# Usage

Firstly, clone this repository by executing in CMD:

```
git clone https://github.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel
```

After cloning the repository, download the required dependencies by running (python >= 3.8):
- For using CPU:

    ```
    pip install -r requirements_cpu.txt
    ```

- For using GPU (sure that you have GPU):

    ```
    pip install -r requirements_gpu.txt
    ```

For command-line usage, execute the following script for inference:

```
python predict.py --img path_image
```

> example: 
    `python .\predict.py --img_path image_test/3.jpg --use_gpu False --draw True`

For Jupyter Notebook, you can explore and experiment with the code at [predict.ipynb](https://github.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel/blob/main/predict.ipynb).

# Pipeline
<div align="center">
  <div style="display: inline-block; text-align: center">
    <img src="https://raw.githubusercontent.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel/refs/heads/main/image_test/3.jpg" width="500" height="600" />
    <p>original image</p>
  </div>
  <div style="display: inline-block; text-align: center;">
    <img src="https://raw.githubusercontent.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel/refs/heads/main/for%20display%20github/box%20words%20level%20detect.png" width="500" height="600" />
    <p>Detect bounding box word level by PaddleOCR</p>
  </div>
</div>
<div align="center" style="margin-top: 40px;">
  <div style="display: inline-block; text-align: center;">
    <img src="https://raw.githubusercontent.com/vietanhlee/Table-Extraction-from-Images-to-CSV-Excel/refs/heads/main/for%20display%20github/csv%20out.png" width="500" height="600" />
    <p>CSV result after recognize by Easyocr and restruct</p>
  </div>
</div>


# References

- [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/quick_start.html)
- [Easyocr](https://github.com/JaidedAI/EasyOCR)