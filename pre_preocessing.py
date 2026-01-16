from PIL import Image,ImageOps
import os

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

os.chdir(r'C:\Users\User\Desktop\DejaView')
image_path=[]

ddirectory="dataset_images"
batches=os.listdir(ddirectory)
os.chdir(ddirectory)
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
                    image_path.append(abs_image_path)
                    print(f'processed image {abs_image_path}')

                os.chdir('..')

def is_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            return True
    except:
        return False

def hash_preprocessing(path):
    img=Image.open(path)
    img=ImageOps.exif_transpose(img)

    if img.mode in ('RGBA','P'):
        img.convert('RGB')
        new_img =Image.new("RGB", img.size, (255, 255, 255))
        new_img.paste(img, mask=img.split()[3])
        return new_img

    return img.convert('RGB')
    
