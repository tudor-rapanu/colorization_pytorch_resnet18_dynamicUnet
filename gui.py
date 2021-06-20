import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import matplotlib.pyplot as plt

from colorizers import *
from colorizers.util import *
from colorizers.own_model import show_color_image

version = '25 May 2021'

w_img_size = 350
h_img_size = 250

# --------------------------------- Helper Functions ---------------------------------
def colorize_image_with_own_model(image_filename=None, cv2_frame=None):
  img = cv2.imread(image_filename) if image_filename else cv2_frame
  out_img_own_model = show_color_image(image_filename)
  plt.imsave('saved_own_model.png', out_img_own_model)
  out_img_own_model = (255 * out_img_own_model).astype("uint8")
  return img, out_img_own_model

def colorize_image_siggraph17(image_filename=None, cv2_frame=None):
  # load colorizers
  colorizer_siggraph17 = siggraph17(pretrained=True).eval()
  #The eval() method parses the expression passed to this method and runs python expression (code) within the program.

  # default size to process images is 256x256
  # grab L channel in both original ("orig") and resized ("rs") resolutions
  img = cv2.imread(image_filename) if image_filename else cv2_frame
  (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

  # colorizer outputs 256x256 ab map
  # resize and concatenate to original L channel

  img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
  out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

  plt.imsave('saved_siggraph17.png', out_img_siggraph17)

  out_img_siggraph17 = (255 * out_img_siggraph17).astype("uint8")
  return img, out_img_siggraph17


def colorize_image_eccv16(image_filename=None, cv2_frame=None):
  # load colorizers
  colorizer_eccv16 = eccv16(pretrained=True).eval()
  #The eval() method parses the expression passed to this method and runs python expression (code) within the program.

  # default size to process images is 256x256
  # grab L channel in both original ("orig") and resized ("rs") resolutions
  img = cv2.imread(image_filename) if image_filename else cv2_frame
  (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))

  # colorizer outputs 256x256 ab map
  # resize and concatenate to original L channel

  img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
  out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

  plt.imsave('saved_eccv16.png', out_img_eccv16)

  out_img_eccv16 = (255 * out_img_eccv16).astype("uint8")
  return img, out_img_eccv16


def convert_to_grayscale(frame):
  gray = cv2.imread(frame)
  (tens_l_orig, tens_l_rs) = preprocess_img(gray, HW=(256,256))
  img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
  return img_bw


def resize_image(image):
  h_img_size = image.shape[0]
  ratio = w_img_size / float(image.shape[1])
  image_resized = cv2.resize(image, (w_img_size, int(h_img_size*ratio)), interpolation = cv2.INTER_AREA)
  return image_resized


# --------------------------------- The GUI ---------------------------------

# First the window layout...2 columns

left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')],
            [sg.Text('Version ' + version, font='Courier 8')]]

images_col = [[sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-'), sg.FileBrowse()],
              [sg.Button('Colorize Photo with Own Model', key='-OWN-'), 
              sg.Button('Colorize Photo with ECCV16', key='-ECCV16-'), 
              sg.Button('Colorize Photo with SIGGRAPH17', key='-SIGGRAPH17-'), 
              sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')],]
# ----- Full layout -----
layout = [[sg.Column(left_col), sg.VSeperator(), sg.Column(images_col)]]

# ----- Make the window -----
window = sg.Window('Photo Colorizer', layout, grab_anywhere=True, finalize=True)

# ----- Run the Event Loop -----
prev_filename = colorized = cap = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        # get list of files in folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            image_resized = resize_image(image)
            window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')
        except:
            continue
    elif event == '-OWN-':        # Colorize photo button clicked
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            image, colorized = colorize_image_with_own_model(filename)
            image_resized = resize_image(image)
            window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
            colorized_saved = cv2.imread(os.path.join(os.getcwd(), 'saved_own_model.png'))
            colorized_resized = resize_image(colorized_saved)
            window['-OUT-'].update(data=cv2.imencode('.png', colorized_resized)[1].tobytes())
        except:
            continue
    elif event == '-ECCV16-':        # Colorize photo button clicked
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            image, colorized = colorize_image_eccv16(filename)
            image_resized = resize_image(image)
            window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
            colorized_saved = cv2.imread(os.path.join(os.getcwd(), 'saved_eccv16.png'))
            colorized_resized = resize_image(colorized_saved)
            window['-OUT-'].update(data=cv2.imencode('.png', colorized_resized)[1].tobytes())
        except:
            continue
    elif event == '-SIGGRAPH17-':        # Colorize photo button clicked
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            image, colorized = colorize_image_siggraph17(filename)
            image_resized = resize_image(image)
            window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
            colorized_saved = cv2.imread(os.path.join(os.getcwd(), 'saved_siggraph17.png'))
            colorized_resized = resize_image(colorized_saved)
            window['-OUT-'].update(data=cv2.imencode('.png', colorized_resized)[1].tobytes())
        except:
            continue
    elif event == '-IN FILE-':      # A single filename was chosen
        filename = values['-IN FILE-']
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)
                image_resized = resize_image(image)
                window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
            except:
                continue
    elif event == '-SAVE-' and colorized is not None:   # Clicked the Save File button
        filename = sg.popup_get_file('Save colorized image.\nColorized image be saved in format matching the extension you enter.', save_as=True)
        try:
            if filename:
                cv2.imwrite(filename, colorized)
                sg.popup_quick_message('Image save complete', background_color='red', text_color='white', font='Any 16')
        except:
            sg.popup_quick_message('ERROR - Image NOT saved!', background_color='red', text_color='white', font='Any 16')
# ----- Exit program -----
window.close()