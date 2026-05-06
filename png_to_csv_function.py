# This function is for the meantime redundant since it's for creating datasets



# import pandas as pd
# import numpy as np
# from PIL import Image
# import os
# from tqdm import tqdm

# def images_to_csv(folder_path, output_csv='letters.csv', img_size=28):
#     data = []
#     labels = []
    
#     for label in tqdm(os.listdir(folder_path)):
#         label_folder = os.path.join(folder_path, label)
#         if not os.path.isdir(label_folder):
#             continue
            
#         for img_file in os.listdir(label_folder):
#             img_path = os.path.join(label_folder, img_file)
#             try:
#                 img = Image.open(img_path).convert('L')      
#                 img = img.resize((img_size, img_size))
#                 pixel_values = np.array(img).flatten()     
#                 data.append(pixel_values)
#                 labels.append(label)
#             except:
#                 continue
    
    
#     df = pd.DataFrame(data)
#     df['label'] = labels
#     df.to_csv(output_csv, index=False)
#     print(f"Saved {len(df)} images to {output_csv}")
    


#images_to_csv("PNGs")
#