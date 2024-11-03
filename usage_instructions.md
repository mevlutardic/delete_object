# Usage Instructions

## 1. Getting Started

To start your project, run the `main.py` file. This will open a window to process images from the specified image files folder.

## 2. Controls

While the application is running, you can use the following keyboard controls:

- **'q'** : Exits the program.
- **'n'** : Moves to the next photo.
- **'b'** : Returns to the previous photo.
- **'m'** : Toggles the drawing mode.
- **'c'** : Enables circle drawing mode.
- **'g'** : Runs the GrabCut algorithm.
- **'r'** : Removes parts of the image.
- **'0'** : Selects background (BG) drawing.
- **'1'** : Selects foreground (FG) drawing.
- **'2'** : Selects probable background (PR_BG) drawing.
- **'3'** : Selects probable foreground (PR_FG) drawing.

## 3. Image Processing

1. **Select an Image**: Choose an image from the `src_image_folder` directory.
2. **Drawing Mode**: Activate the background or foreground drawing mode and draw over the object you want to remove using the mouse.
3. **GrabCut Algorithm**: Once your drawing is complete, press 'g' to run the GrabCut algorithm.
4. **Inpainting**: Press 'r' to remove the selected areas. The result will be displayed in a new image.
5. **Saving Results**: If desired, you can save the processed image by pressing 's'.

## 4. Notes

- In drawing mode, the blue outline indicates the background, the white fill indicates the foreground, and red and green represent probable background and foreground respectively.
- The GrabCut algorithm can be run multiple times to achieve better results.
