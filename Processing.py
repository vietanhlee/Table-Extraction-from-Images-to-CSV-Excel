import cv2
import numpy as np
import easyocr
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class Processing:
    def __init__(self, num_threads=4, gpu = False, lang = 'en'):
        # For task recogineze
        self.lang = lang
        if lang == 'vi':
            self.reader = easyocr.Reader(['vi', 'en'], verbose=False, gpu= gpu)
        elif lang != 'en':
            raise Exception(f"Language '{lang}' is not supported. Only 'vi'or 'en' is supported.")
        # For task detect
        self.paddle_reader = PaddleOCR(lang='en', show_log=False,use_gpu = gpu)
        # For run many task simultaneously (don't test this function, it is not complete) 
        self.num_threads = num_threads

    def find_rects_texts_easyocr(self, img_path, mode_draw= 0):
        '''Function for find rects (bounding box) and text respectively'''
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at path: {img_path}")
        img = image.copy()
        a = 10
        b = a / 1.5
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (int(w * a), int(h * b)))
        result = self.paddle_reader.ocr(image_resized, cls=True, rec= False)[0][::-1]

        rects, horizontal_list, texts = [], [], []

        padding = 2
        for line in result:
            box = np.array(line, dtype=np.float32)
            x, y, w, h = cv2.boundingRect(box)
            x1, y1 = max(int(int(x / a) - padding), 0), max(int(int(y / b) - padding), 0)
            x2, y2 = int(int((x + w) / a) + padding), int(int((y + h) / b) + padding)

            rects.append([x1, y1, x2, y2])
            horizontal_list.append([x1, x2, y1, y2])

        rects = np.array(rects, dtype=int)
        results = self.reader.recognize(img, horizontal_list=horizontal_list, free_list=[])

        for _, text, _ in results:
            text = text.strip()
            if text == '1.': text = 'I.'
            elif text == '11.': text = 'II.'
            elif len(text) >= 2 and text[0] == '[' and text[-1].isdigit():
                text = '1'+ text[1:]
            else:
                chars = list(text)
                for i in range(1, len(chars)):
                    if chars[i] == '1' and chars[i - 1] == 'I':
                        chars[i] = 'I'
                text = ''.join(chars)
            texts.append(text)

        if mode_draw:
            for rect in rects:
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
            plt.figure(figsize=(12, 12))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return rects, texts
    def find_rects_texts_paddle_all(self, img_path, mode_draw= 0):
        '''Function for find rects (bounding box) and text respectively'''
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at path: {img_path}")
        img = image.copy()
        a = 10
        b = a / 1.5
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (int(w * a), int(h * b)))
        result = self.paddle_reader.ocr(image_resized, cls=True, rec= True)[0]

        rects, horizontal_list, texts = [], [], []

        padding = 2
        for line in result:
            box = np.array(line[0], dtype=np.float32)
            
            x, y, w, h = cv2.boundingRect(box)
            
            x1, y1 = max(int(int(x / a) - padding), 0), max(int(int(y / b) - padding), 0)
            x2, y2 = int(int((x + w) / a) + padding), int(int((y + h) / b) + padding)

            rects.append([x1, y1, x2, y2])
            
            text = line[1][0]
            text = text.strip()
            if text == '1.': text = 'I.'
            elif text == '11.': text = 'II.'
            elif len(text) >= 2 and text[0] == '[' and text[-1].isdigit():
                text = '1'+ text[1:]
            else:
                chars = list(text)
                for i in range(1, len(chars)):
                    if chars[i] == '1' and chars[i - 1] == 'I':
                        chars[i] = 'I'
                text = ''.join(chars)
            texts.append(text)
            
        rects = np.array(rects, dtype=int)

        if mode_draw:
            for rect in rects:
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
            plt.figure(figsize=(12, 12))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return rects, texts
        
    def check_line(self, box1, box2):
        '''Check two bounding are all on a line'''
        return abs((box1[1] + box1[3]) // 2 - (box2[1] + box2[3]) // 2) < 10

    def rects_texts_ncollum_processed(self, rects, texts):
        rects_new, txts_new = [], []
        current_rect, current_txt = [], []
        max_col = 0

        for i, text in enumerate(texts):
            text = text.strip()
            if not text: 
                continue
            if text[0].isdigit() and text[-1].isdigit():
                text = text.replace(',', '.')

            if not current_rect or self.check_line(current_rect[-1], rects[i]):
                current_rect.append(rects[i])
                current_txt.append(text)
            else:
                combined = sorted(zip(current_rect, current_txt), key=lambda x: (x[0][0] + x[0][2]) // 2)
                cur_rect_sorted, cur_txt_sorted = zip(*combined)
                rects_new.append(np.array(cur_rect_sorted))
                txts_new.append(list(cur_txt_sorted))
                max_col = max(max_col, len(cur_txt_sorted))
                current_rect = [rects[i]]
                current_txt = [text]

        if current_rect:
            combined = sorted(zip(current_rect, current_txt), key=lambda x: (x[0][0] + x[0][2]) // 2)
            cur_rect_sorted, cur_txt_sorted = zip(*combined)
            rects_new.append(np.array(cur_rect_sorted))
            txts_new.append(list(cur_txt_sorted))
            max_col = max(max_col, len(cur_txt_sorted))

        return rects_new, txts_new, max_col

    def find_box_cols(self, rects_box, n_cols):
        cols_data = [row[:, [0, 2]].flatten() for row in rects_box if len(row) == n_cols]
        cols_data = np.array(cols_data)

        col_xmin = cols_data[:, ::2].min(axis=0)
        col_xmax = cols_data[:, 1::2].max(axis=0)

        return np.stack((col_xmin, col_xmax), axis=1)

    def find_text_each_row(self, box_cols, list_rects, list_texts):
        result = []
        for row_rects, row_texts in zip(list_rects, list_texts):
            row_data = [''] * len(box_cols)
            for rect, text in zip(row_rects, row_texts):
                x_center = (rect[0] + rect[2]) // 2
                for idx, (xmin, xmax) in enumerate(box_cols):
                    if xmin <= x_center <= xmax:
                        row_data[idx] = (row_data[idx] + ' ' + text).strip() if row_data[idx] else text
                        break
            result.append(row_data)
        return result

    def process_single_image(self, img_path, draw=0):
        if self.lang == 'vi':
            rects, texts = self.find_rects_texts_easyocr(img_path, draw)
        else:
            rects, texts = self.find_rects_texts_paddle_all(img_path, draw)
                
        rects_grouped, texts_grouped, n_cols = self.rects_texts_ncollum_processed(rects, texts)
        box_cols = self.find_box_cols(rects_grouped, n_cols)
        return self.find_text_each_row(box_cols, rects_grouped, texts_grouped)
    
    def processing(self, img_paths):
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        results_all = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self.process_single_image, path, draw = 0): path
                for path in img_paths
            }

            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    result = future.result()
                    results_all.append(result)

                    # Nếu bật chế độ xuất CSV
                    df = pd.DataFrame(result)
                    out_file = os.path.splitext(img_path)[0] + '.csv'
                    df.to_csv(out_file, index=False, header=False)
                    print(f"✅ Output CSV saved to: {out_file}")

                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
        
        return results_all