from PIL import Image,ImageOps
import os
from imagehash import ImageHash


pdirectory="preprocessed_hash_images"

try:
    os.mkdir(pdirectory)
    print(f"Directory made:{pdirectory}")
    fullPath = os.path.abspath(pdirectory)
    print(f"Folder created at: {fullPath}")
    
except FileExistsError:
    print(f"Directory '{pdirectory}' already exists")
    
except PermissionError:
    print("Permission denied.")


os.chdir(r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing\preprocessed_hash_images')
image_path=[]
hash_values=[]
batches=['batch_1']


ddirectory=r'C:\Users\Tarush Banke\OneDrive\Desktop\Turing Playground\storage\preproccessing\dataset_images'
os.chdir(ddirectory)


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



for batch in batches:
    full_path = os.path.abspath(batch)
    if(os.path.isdir(full_path)):
        os.chdir(full_path)

        sub_batch=os.listdir(os.getcwd())
        for sub in sub_batch:
            if(os.path.isdir(full_path+'/'+sub)):
                os.chdir(full_path+'/'+sub)
                print(os.getcwd())

                for image in os.scandir():
                    abs_image_path=os.path.abspath(image)
                    if not is_image(abs_image_path):
                        continue
                    image_path.append(abs_image_path)
                    print(f'processed image {abs_image_path}')

                os.chdir('..')



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
    


