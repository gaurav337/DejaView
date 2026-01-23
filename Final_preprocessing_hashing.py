from PIL import Image,ImageOps
import os
import imagehash 
import json
import numpy as np

def is_image(path):
    size=os.path.getsize(path)
    if size==0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except:
        return False

def pw_hash(path) :
    img=Image.open(path)
    p_hash=imagehash.phash(img)
    w_hash=imagehash.whash(img)
    return p_hash,w_hash


def hash_preprocessing(path):
    img=Image.open(path)
    img=ImageOps.exif_transpose(img)

    if img.mode in ('RGBA','LA'):
        new_img =Image.new("RGB", img.size, (255, 255, 255))
        new_img.paste(img, mask=img.split()[3])
        img=new_img
    else:
        img=img.convert('RGB')
        
    return img


def alter_image(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')
    
    return {
        "original": img,
        "rotate_90": img.rotate(270, expand=True),
        "rotate_180": img.rotate(180, expand=True),
        "rotate_270": img.rotate(90, expand=True),
        "flip_horizontal": img.transpose(Image.FLIP_LEFT_RIGHT),
        "flip_vertical": img.transpose(Image.FLIP_TOP_BOTTOM),
        "flip_both": img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT),
    }


def get_augmented_hashes(image_path):
    augmented_images = alter_image(image_path)
    hashes = []
    
    for aug_name, img in augmented_images.items():
        p_hash = imagehash.phash(img)
        w_hash = imagehash.whash(img)
        hashes.append((p_hash, w_hash, aug_name))
    
    return hashes


# if __name__ == "__main__":
#     pdirectory="preprocessed_hash_images"
#     try:
#         os.mkdir(pdirectory)
#         print(f"Directory made:{pdirectory}")
#         fullPath = os.path.abspath(pdirectory)
#         print(f"Folder created at: {fullPath}")
#     except FileExistsError:
#         print(f"Directory '{pdirectory}' already exists")
#     except PermissionError:
#         print("Permission denied.")

#     os.chdir(r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing\preprocessed_hash_images')
#     image_path=[]
#     hash_values=[]
#     batches=['batch_1']

#     ddirectory=r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing\dataset_images'
#     os.chdir(ddirectory)


#     for batch in batches:
#         full_path = os.path.abspath(batch)
#         if(os.path.isdir(full_path)):
#             os.chdir(full_path)

#             sub_batch=os.listdir(os.getcwd())
#             for sub in sub_batch:
#                 if(os.path.isdir(full_path+'/'+sub)):
#                     os.chdir(full_path+'/'+sub)
#                     print(os.getcwd())

#                     for image in os.scandir():
#                         abs_image_path=os.path.abspath(image)
#                         if not is_image(abs_image_path):
#                             continue
#                         image_path.append(abs_image_path)
#                         print(f'processed image {abs_image_path}')

#                     os.chdir('..')
                    

#     preprocessed_folder_path=r"storage/preproccessing/preprocessed_images/batch_1"
#     image_name=os.path.basename(image_path[0])
#     image_name_png= image_name.replace(".jpg", ".png").replace(".jpeg", ".png")
#     full_save_path = os.path.join(preprocessed_folder_path,image_name_png)

#     preprocessed_folder_path=r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing\preprocessed_images\batch_1'
#     counter=1

#     for image in image_path :
#         image_name=os.path.basename(image)
#         image_name_png= image_name.replace(".jpg", ".png").replace(".jpeg", ".png")
#         full_save_path = os.path.join(preprocessed_folder_path,image_name_png)
        
#         img = hash_preprocessing(image)
#         img.save(full_save_path)
        
#         print(f"dir {img} {counter} {full_save_path}")
#         counter+=1
        

#     folder_name="C:\\Users\\Tarush Banke\\OneDrive\\Desktop\\Turing Playground\\storage\\preproccessing\\preprocessed_images"
#     for batch in batches:
#         batch_path=os.path.join(folder_name,batch)
#         print(batch_path)
#         images=os.scandir(batch_path)
        
#         for image in images:
#             if image.is_file():
#                 if is_image(image):
#                     p_hash,w_hash=pw_hash(image)
#                     hash_values.append({
#                         'name': image.name,
#                         'phash': p_hash, 
#                         'whash': w_hash
#                     })
#                     print(f'hashed the image :{image.name}---phash:{p_hash}---whash:{w_hash}')



#     json_data=[]

#     for entry in hash_values:
#         new_entry={
#             'name':entry['name']
#         }

#         if isinstance(entry['phash'],np.ndarray):
#             new_entry['phash'] = str(imagehash.ImageHash(entry['phash']))
#         else:
#             new_entry['phash'] = str(entry['phash'])
        
#         if isinstance(entry['whash'], np.ndarray):
#             new_entry['whash'] = str(imagehash.ImageHash(entry['whash']))
#         else:
#             new_entry['whash'] = str(entry['whash'])
            
#         json_data.append(new_entry)
#         print(new_entry)


#     os.chdir(r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing')

#     with open('image_hashes.json', 'w') as f:
#         json.dump(json_data, f, indent=4)

#     print(f"Successfully converted and saved {len(json_data)} hashes!")
#     print("Example output:", json_data[0])

