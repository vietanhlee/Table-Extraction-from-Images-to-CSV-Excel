from Processing import Processing
import pandas as pd
import argparse

def main(gpu = False, path = r'image test\7.jpg', lang = 'en', draw = 0):
    
    tool = Processing(gpu= gpu, lang= lang)
    DF = tool.process_single_image(img_path= path, draw= draw)
    
    if draw:
        print('Image table detected was displayed with bounding box words level')
        print('Please close matplotlib window to display Dataframe CSV output\n')
    
    print(pd.DataFrame(DF))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True, help= 'path of image for OCR')
    parser.add_argument('--use_gpu', required=False, default= "False", help= 'is use GPU?')
    parser.add_argument('--draw', required=False, default= 'False', help= 'is display image after draw bounding box word level?')
    parser.add_argument('--lang', required= False, default= 'en', help= 'set language for OCR, default is en for english, you can set vi for vietnamese ')
    args = parser.parse_args()

    # Ép kiểu về từ string về bool
    args.use_gpu = args.use_gpu.lower() in ['true', '1', 'yes']
    args.draw = args.draw.lower() in ['true', '1', 'yes']
    
    main(args.use_gpu, args.img_path, args.lang, args.draw)