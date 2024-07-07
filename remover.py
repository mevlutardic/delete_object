import cv2
import os
import numpy as np
import threading
from skimage.restoration import inpaint


class GrabCut(threading.Thread):
    def __init__(self, windowName, filepath, kSize):
        threading.Thread.__init__(self)

        self.windowName = windowName
        self.filepath = filepath
        self.kSize = kSize
        self.i = 0
        self.sourcelist = os.listdir(self.filepath)

        cv2.namedWindow(self.windowName, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)

        cv2.setMouseCallback(self.windowName, self.drawing_process)

        self.mode = False

        path = os.path.join(self.filepath, self.sourcelist[self.i])
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        self.resize_orj_img = cv2.resize(self.image, (int(self.image.shape[1] * self.kSize), int(self.image.shape[0] * self.kSize)))

        self.image_new = self.resize_orj_img.copy()

        self.mask = np.zeros(self.image_new.shape[:2], dtype=np.uint8)
        
        self.mask2 = None

        self.output = np.zeros(self.image_new.shape, np.uint8)

        cv2.imshow(self.windowName, self.image_new)
        

        self.BLUE = [255, 0, 0]
        RED = [0, 0, 255]
        GREEN = [0, 255, 0]
        BLACK = [0, 0, 0]
        WHITE = [255, 255, 255]

        self.DRAW_BG = {'color': BLACK, 'val': 0}
        self.DRAW_FG = {'color': WHITE, 'val': 1}
        self.DRAW_PR_FG = {'color': GREEN, 'val': 3}
        self.DRAW_PR_BG = {'color': RED, 'val': 2}

        self.rect = (0, 0, 1, 1)
        self.drawing = False
        self.rectangle = False
        self.rect_over = False
        self.value = self.DRAW_FG
        self.rect_or_mask = None
        self.start_point_rect = (0, 0)
        self.thickness = 3

        self.draw_circle = False

        self.img = None
        
    def informations(self):
        print("""
        Press 'q' to exit the loop
        Press 'n' to go to the next photo
        Press 'b' to go to the previous photo
        Press 'm' to toggle mode (e.g., exit drawing mode)
        Press 'c' to enable circle drawing mode
        Press 'g' to run the GrabCut algorithm
        
        """)

    def run(self):
        while True:
            if not self.mode and not self.draw_circle:
                cv2.imshow(self.windowName, self.image_new)

            cv2.imshow('output', self.output)

            code = cv2.waitKey(1)

            if code == ord("q"):  # Press 'q' to exit the loop
                print("Exiting the program...")
                break
            elif code == ord("n"):  # Press 'n' to go to the next photo
                if self.i < len(self.sourcelist) - 1:
                    self.i += 1
                    self.update_photo(self.i)
                    print("Moving to the next photo...")
            elif code == ord("b"):  # Press 'b' to go to the previous photo
                if self.i > 0:
                    self.i -= 1
                    self.update_photo(self.i)
                    print("Moving to the previous photo...")
            elif code == ord("m"):  # Press 'm' to toggle mode (e.g., exit drawing mode)
                self.mode = not self.mode
                self.draw_circle = False
                print("Toggling mode...")
            elif code == ord("c"):  # Press 'c' to enable circle drawing mode
                self.draw_circle = True
                print("Circle drawing mode enabled...")
            elif code == ord("g"):  # Press 'g' to run the GrabCut algorithm
                self.grab_cut()
                print("Running GrabCut algorithm...")
            elif code == ord("r"):  # Press 'r' to remove parts of the image
                kernel = np.ones((5,5), np.uint8)
                self.mask2 = cv2.dilate(self.mask2, kernel, iterations=1)
                dst = cv2.inpaint(self.image_new, self.mask2, 3, cv2.INPAINT_NS)
                filled_image = inpaint.inpaint_biharmonic(self.image_new, self.mask2, split_into_regions=True, channel_axis=-1)
                filled_image2 = inpaint.inpaint_biharmonic(dst, self.mask2, split_into_regions=True, channel_axis=-1)
                cv2.imshow("out", dst)
                cv2.imshow("filled_image", filled_image)
                cv2.imshow("filled_image2", filled_image2)
                print("Removing parts of the image and displaying results...")
            elif code == ord('0'):  # Press '0' for background (BG) drawing
                self.value = self.DRAW_BG
                print("Background drawing selected...")
            elif code == ord('1'):  # Press '1' for foreground (FG) drawing
                self.value = self.DRAW_FG
                print("Foreground drawing selected...")
            elif code == ord('2'):  # Press '2' for probable background (PR_BG) drawing
                self.value = self.DRAW_PR_BG
                print("Probable background drawing selected...")
            elif code == ord('3'):  # Press '3' for probable foreground (PR_FG) drawing
                self.value = self.DRAW_PR_FG
                print("Probable foreground drawing selected...")


        cv2.destroyAllWindows()
            
    def grab_cut(self):
        if self.rect_or_mask == 0:  # grabcut with rect
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            self.img = cv2.cvtColor(self.image_new, cv2.COLOR_RGBA2RGB)
            cv2.grabCut(self.img, self.mask, self.rect, bgdmodel, fgdmodel, 7, cv2.GC_INIT_WITH_RECT)
            self.rect_or_mask = 1
        elif self.rect_or_mask == 1:  # grabcut with mask
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            self.img = cv2.cvtColor(self.image_new, cv2.COLOR_RGBA2RGB)
            cv2.grabCut(self.img, self.mask, self.rect, bgdmodel, fgdmodel, 7, cv2.GC_INIT_WITH_MASK)
        
        self.mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
        
        if np.any(self.mask2): 
            
            self.output = cv2.bitwise_and(self.img, self.img, mask=self.mask2)
           
        else:
            print("mask2 is empty.")

    def update_photo(self, i):
        path = os.path.join(self.filepath, self.sourcelist[i])
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.image = cv2.resize(self.image, (int(self.image.shape[1] * self.kSize), int(self.image.shape[0] * self.kSize)))
        self.image_new = self.image.copy()
        self.mask = np.zeros(self.image_new.shape[:2], dtype=np.uint8)
        self.output = np.zeros(self.image_new.shape, np.uint8)

    def drawing_process(self, event, x, y, flags, param):
        if self.mode and not self.draw_circle:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.rectangle = True
                self.start_point_rect = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectangle:
                    self.image2 = self.image_new.copy()
                    ix = self.start_point_rect[0]
                    iy = self.start_point_rect[1]
                    overlay = self.image2.copy() 
                    cv2.rectangle(self.image2, (ix, iy), (x, y), self.BLUE, 3)
                    alpha = 0.6
                    self.image2 = cv2.addWeighted(overlay, alpha, self.image2, 1 - alpha, 0) 
                    self.rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
                    self.rect_or_mask = 0
                    cv2.imshow(self.windowName, self.image2)
            elif event == cv2.EVENT_LBUTTONUP:
                self.rectangle = False
                self.rect_over = True
                ix = self.start_point_rect[0]
                iy = self.start_point_rect[1]
                self.rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
                self.rect_or_mask = 0

        elif self.rect_over and self.draw_circle:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                cv2.circle(self.image2, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    cv2.circle(self.image2, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)
                    cv2.imshow(self.windowName, self.image2)
            elif event == cv2.EVENT_LBUTTONUP:
                if self.drawing:
                    self.drawing = False
                    cv2.circle(self.image2, (x, y), self.thickness, self.value['color'], -1)
                    cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

if __name__ == '__main__':
    photo_path = "/home/mard/Desktop/grabcut/images/"
    kSize = 1.1
    thread = GrabCut("Window", photo_path, kSize)
    thread.run()