import modules.scripts as scripts
import gradio as gr
import os
import torch

from modules.processing import process_images
from modules.paths import models_path
from modules.textual_inversion import autocrop
import cv2
import copy
import numpy as np
from PIL import Image
import time
import requests


def get_my_dir():
    if os.path.isdir("extensions/face_crop_img2img"):
        return "extensions/face_crop_img2img"
    return scripts.basedir()

def x_ceiling(value, step):
    return -(-value // step) * step

def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

def download_and_cache_models(dirname):
    download_url = 'https://github.com/zymk9/yolov5_anime/blob/8b50add22dbd8224904221be3173390f56046794/weights/yolov5s_anime.pt?raw=true'
    model_file_name = 'yolov5s_anime.pt'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cache_file = os.path.join(dirname, model_file_name)
    if not os.path.exists(cache_file):
        print(f"downloading face detection model from '{download_url}' to '{cache_file}'")
        response = requests.get(download_url)
        with open(cache_file, "wb") as f:
            f.write(response.content)

    if os.path.exists(cache_file):
        return cache_file
    return None


class Script(scripts.Script):
    anime_face_detector = None
    face_detector = None
    mask_file = "face_crop_img2img_mask.png"
    mask_image = None

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "face crop img2img"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        face_detection_method = gr.Dropdown(choices=["YuNet","Yolov5_anime"], value="YuNet" ,label="Face Detection Method")
        gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                If loading of the Yolov5_anime model fails, check\
                 <font color=\"blue\"><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2235\">[this]</a></font> solution.\
                </p>")
        max_crop = gr.Slider(minimum=0, maximum=2048, step=1, value=1024, label="Max Crop Size")
        face_denoising_strength = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.5, label="Face Denoising Strength")
        face_area = gr.Slider(minimum=1.00, maximum=3.00, step=0.01, value=1.5, label="Face Area Magnification")

        with gr.Column():
            enable_face_prompt = gr.Checkbox(False, label="Enable Face Prompt")
            face_prompt = gr.Textbox(label="Face Prompt", show_label=False, lines=2,
                placeholder="Prompt for Face",
                value = "face close up,"
            )

        return [face_detection_method, max_crop, face_denoising_strength,face_area,enable_face_prompt,face_prompt]

    
    def detect_face(self, img_array):
        if not self.face_detector:
            dnn_model_path = autocrop.download_and_cache_models(os.path.join(models_path, "opencv"))
            self.face_detector = cv2.FaceDetectorYN.create(dnn_model_path, "", (0, 0))
        
        self.face_detector.setInputSize((img_array.shape[1], img_array.shape[0]))
        _, result = self.face_detector.detect(img_array)
        return result

    def detect_anime_face(self, img_array):
        if not self.anime_face_detector:
            anime_model_path = download_and_cache_models(os.path.join(models_path, "yolov5_anime"))

            if not os.path.isfile(anime_model_path):
                print( "WARNING!! " + anime_model_path + " not found.")
                print( "use YuNet instead.")
                return self.detect_face(img_array)
            
            self.anime_face_detector = torch.hub.load('ultralytics/yolov5', 'custom', path=anime_model_path)
        
        result = self.anime_face_detector(img_array)
        #models.common.Detections
        faces = []
        for x_c, y_c, w, h, _, _ in result.xywh[0].tolist():
            faces.append( [ x_c - w/2 , y_c - h/2, w, h ] )
        
        return faces

    def get_mask(self):
        def create_mask( output, x_rate, y_rate, k_size ):
            img = np.zeros((512, 512, 3))
            img = cv2.ellipse(img, ((256, 256), (int(512 * x_rate), int(512 * y_rate)), 0), (255, 255, 255), thickness=-1)
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)
            cv2.imwrite(output, img)
        
        if self.mask_image is None:
            mask_file_path = os.path.join( get_my_dir() , self.mask_file)
            if not os.path.isfile(mask_file_path):
                create_mask( mask_file_path, 0.9, 0.9, 91)

            m = cv2.imread( mask_file_path )[:,:,0]
            m = m[:, :, np.newaxis]
            self.mask_image = m / 255
        
        return self.mask_image
    
  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, face_detection_method, max_crop, face_denoising_strength, face_area, enable_face_prompt, face_prompt):

        def img_crop( img, face_coords,face_area,max_crop):
            img_array = np.array(img)
            face_imgs =[]
            new_coords = []

            for face in face_coords:
                x = int(face[0])
                y = int(face[1])
                w = int(face[2])
                h = int(face[3])
                print([x,y,w,h])

                if max(w,h) > max_crop:
                    print("ignore big face")
                    continue

                cx = x + int(w/2)
                cy = y + int(h/2)

                x = cx - int(w*face_area / 2)
                x = x if x > 0 else 0
                w = cx + int(w*face_area / 2) - x
                w = w if x+w < img.width else img.width - x

                y = cy - int(h*face_area / 2)
                y = y if y > 0 else 0
                h = cy + int(h*face_area / 2) - y
                h = h if y+h < img.height else img.height - y

                print([x,y,w,h])

                face_imgs.append( img_array[y: y+h, x: x+w] )
                new_coords.append( [x,y,w,h] )
            
            resized = []
            for face_img in face_imgs:
                if face_img.shape[1] < face_img.shape[0]:
                    re_w = 512
                    re_h = int(x_ceiling( (512 / face_img.shape[1]) * face_img.shape[0] , 64))
                else:
                    re_w = int(x_ceiling( (512 / face_img.shape[0]) * face_img.shape[1] , 64))
                    re_h = 512
                face_img = resize_img(face_img, re_w, re_h)
                resized.append( Image.fromarray(face_img))

            return resized, new_coords


        def merge_face(img, face_img, face_coord, base_img_size, mask):
            x_rate = img.width / base_img_size[0]
            y_rate = img.height / base_img_size[1]

            img_array = np.array(img)
            x = int(face_coord[0] * x_rate)
            y = int(face_coord[1] * y_rate)
            w = int(face_coord[2] * x_rate)
            h = int(face_coord[3] * y_rate)

            face_array = np.array(face_img)
            face_array = resize_img(face_array, w, h)

            mask = resize_img(mask, w, h)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

            bg = img_array[y: y+h, x: x+w]

            img_array[y: y+h, x: x+w] = mask * face_array + (1-mask)*bg

            return Image.fromarray(img_array)
        
        def detect_face(img, mask, face_detection_method):
            img_array = np.array(img)

            if mask is not None:
                mask_array = np.array(mask)/255

                if mask_array.ndim == 2:
                    mask_array = mask_array[:, :, np.newaxis]
                
                img_array = mask_array * img_array
                img_array = img_array.astype(np.uint8)
            
            # image without alpha
            img_array = img_array[:,:,:3]

            if face_detection_method == "YuNet":
                return self.detect_face(img_array)
            elif face_detection_method == "Yolov5_anime":
                return self.detect_anime_face(img_array)

        def save_image(img, dir_path):
            filename = "/" + "face_crop_img2img_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
            cv2.imwrite(dir_path + filename, np.array(img)[:, :, ::-1])

        ### face detect in base img
        base_img = p.init_images[0]

        if base_img is None:
            print("p.init_images[0] is None")
            return process_images(p)

        base_img_size = (base_img.width, base_img.height)

        face_coords = detect_face(base_img, p.image_mask,face_detection_method)

        if face_coords is None or len(face_coords) == 0:
            print("no face detected")
            return process_images(p)

        print(face_coords)
        face_imgs, new_coords = img_crop(base_img, face_coords, face_area, max_crop)

        if not face_imgs:
            return process_images(p)

        face_p = copy.copy(p)

        ### img2img base img
        proc = process_images(p)


        ### img2img for each face
        face_img2img_results = []

        for face, coord in zip(face_imgs, new_coords):
            # cv2.imwrite("scripts/face.png", np.array(face)[:, :, ::-1])
            face_p.init_images = [face]
            face_p.width = face.width
            face_p.height = face.height
            face_p.denoising_strength = face_denoising_strength

            if enable_face_prompt:
                face_p.prompt = face_prompt
            else:
                face_p.prompt = "close-up face ," + face_p.prompt

            if p.image_mask is not None:
                x,y,w,h = coord
                face_p.image_mask = Image.fromarray( np.array(p.image_mask)[y: y+h, x: x+w] )
            
            face_proc = process_images(face_p)
            face_img2img_results.append((face_proc.images[0], coord))
        
        ### merge faces
        bg = proc.images[0]
        mask = self.get_mask()

        for face_img, coord in face_img2img_results:
            bg = merge_face(bg, face_img, coord, base_img_size, mask)
        
        save_image(bg, p.outpath_samples)

        proc.images[0] = bg

        return proc
