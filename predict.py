from Processing import Processing
import pandas as pd

path = r'G:\Table Extraction from Images to CSV Excel\image test\7.jpg'

tool = Processing(gpu= False)
DF = tool.process_single_image(path, draw= 0)
print(pd.DataFrame(DF))


