from Processing import Processing
import pandas as pd

path = r'image test\7.jpg'

tool = Processing(gpu= False)
DF = tool.process_single_image(path, draw= 0)
print(pd.DataFrame(DF))

