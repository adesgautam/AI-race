
import numpy as np
import cv2
import time
import mss
import mss.tools
from PIL import Image
# import keyboard
from pynput import keyboard

 
monitor = {"top": 30, "left": 50, "width": 650, "height": 550}

# , keyboard.KeyCode.from_char('d')
COMBINATION = { keyboard.KeyCode.from_char('a'),  
                keyboard.Key.left, keyboard.Key.right}
current = set()
i=0
up, left, right, onlyup = 0, 0, 0, 0

try:
    training_data = open("training_data1.csv", 'r')
    val = training_data.read().split('\n')[-2]
    print(val)
    i = int(val.split(',')[0].split('.')[0].split('/')[1]) + 1
    print("i:", i)
except Exception as e:
    print(e)


def on_press(key):
    # try:
        # print('alphanumeric key {0} pressed'.format(key.char))
    if key == keyboard.Key.esc:
        listener.stop()

    global training_data, up, left, right, onlyup

    if key in COMBINATION:
        current.add(key)

    val = ''
    for k in current:
        if k == keyboard.KeyCode.from_char('a'):
            val += "UP "
            up += 1
        elif k == keyboard.Key.left:
            val += "LEFT "
            left += 1
        elif k == keyboard.Key.right:
            val += "RIGHT "
            right += 1

    if keyboard.KeyCode.from_char('a') in current and len(current)==1:
        onlyup += 1

    if val != '':
        img, path = get_image()
        training_data = open("training_data1.csv", 'a')
        training_data.write(path + ", " + val + "\n")

    print("i:", i, "Written:", val, "ONLYUP:", onlyup, "UP:", up, "LEFT:", left, "RIGHT:", right)

    # if key == keyboard.Key.esc:
    #     training_data.close()
    # except AttributeError:
    #     print('special key {0} pressed'.format(key))

def on_release(key):
    # print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        training_data.close()
        return False
    try:
        current.remove(key)
    except KeyError:
        pass
    
def get_image():
    with mss.mss() as sct:
        global i
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = img[270:650, 0:] # crop

        path = "pics1/"+str(i)+".jpg"
        cv2.imwrite(path, img)
        # mss.tools.to_png(sct_img.rgb, sct_img.size, output=path)
        
        i+=1
        return img, path

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

























# i=0
# while(True):
#     last_time = time.time()
#     img = get_image()
#     # key = get_key()
#     # if i%2==0:
#     print("Pressed", key, i)
#     # print('loop took {} seconds'.format(time.time()-last_time))
#     last_time = time.time()

#     # cv2.imshow("Image", img)
#     # cv2.imwrite("pics/"+i+'.png',img)

#     # if cv2.waitKey(25) & 0xFF == ord('q'):
#     #     cv2.destroyAllWindows()
#     #     break

#     i+=1
    
    




### OLD METHOD (took 0.2xx seconds)
# import numpy as np
# from PIL import ImageGrab
# import cv2
# import time
# import pyautogui

# def screen_record(): 
#     last_time = time.time()
#     while(True):
#         # pyautogui.typewrite('w') 
#         printscreen =  np.array(ImageGrab.grab(bbox=(0,40,600,400)))
#         print('loop took {} seconds'.format(time.time()-last_time))
#         last_time = time.time()
#         cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

# screen_record()

