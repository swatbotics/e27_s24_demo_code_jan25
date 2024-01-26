import cv2
import numpy as np

######################################################################
# a helper function to put an image on the screen

def show_image(img, text):

    # make a copy of the image so we can draw text on it
    tmp = img.copy()

    # get the height of the image
    h = img.shape[0]

    # draw black outline then white text
    for color, thickness in [(0, 5), (255, 1)]:

        # draw text onto temp image of given color and thickness
        cv2.putText(tmp, text, 
                    (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (color, color, color),
                    thickness, cv2.LINE_AA)

    # show our image in a window named 'hello'
    cv2.imshow('hello', tmp)

    # note you could replace this entire while loop with
    # cv2.waitKey() with no args, but then you might 
    # not be able to interrupt the program with Ctrl + C

    while True:
        # wait for 5 ms to get a key press and
        # return >= 0 if pressed
        k = cv2.waitKey(5)

        # if key hit break out of this loop
        if k >= 0:
            break
    
######################################################################
# our main function

def main():

    ############################################################

    # image from https://commons.wikimedia.org/wiki/File:Osteospermum_%22Cape_Daisy_Mary%22.JPG
    img = cv2.imread('osteospermum.jpg')

    # images are represented using numpy arrays of dtype np.uint8
    # color (BGR) images have shape (h, w, 3)
    print(f'img shape is {img.shape}, data type is {img.dtype}')

    show_image(img, 'original image')

    ############################################################

    # note that we want to make a copy here otherwise subsequent
    # img uses will also have a blue corner (try deleting 
    # the .copy() and see what happens in the program)
    img_blue_corner = img.copy()

    # we can use array indexing with numpy to set individual pixels
    img_blue_corner[:100, :100] = (255, 0, 0)

    # show blue corner image
    show_image(img_blue_corner, 'now with a blue corner')

    # we can use OpenCV functions to do color conversions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray images have shape (h, w) 
    print(f'gray shape is {gray.shape}, data type is {gray.dtype}')

    show_image(gray, 'converted to grayscale')

    ############################################################
    
    # we can use indexing to flip an image upside down and backwards
    # (180 deg rotation about center)
    img_flipped = img[::-1, ::-1]

    show_image(img_flipped, 'flipped')

    ############################################################
    # we can use OpenCV functions to do image filtering operations
    # like blurring

    img_blur_a_bit = cv2.GaussianBlur(img, (0, 0), 2.0)

    show_image(img_blur_a_bit, 'blur a bit')

    img_blur_a_lot = cv2.GaussianBlur(img, (0, 0), 20.0)

    show_image(img_blur_a_lot, 'blur a lot')

    ############################################################
    # you have to be careful of overflow when doing arithmetic
    # (e.g. averaging) on images 

    avg1 = (img_flipped + img_blur_a_lot) // 2

    show_image(avg1, 'images averaged incorrectly')

    avg2 = img_flipped // 2 + img_blur_a_lot // 2

    show_image(avg2, 'images averaged correctly using integer math')

    avg3 = ((img_flipped.astype(np.float32) +
            img_blur_a_lot.astype(np.float32)) / 2).astype(np.uint8)

    show_image(avg3, 'images averaged correctly using floating-point math')

    ############################################################
    # demonstrate some neat color thresholding operations

    # petal color as revealed by digital color meter
    petal_color = (235, 94, 221)

    # pixel-wise difference in BGR space from petal color
    # every pixel in diff_from_petal_color holds (delta_B, delta_G, delta_R)
    diff_from_petal_color = (img.astype(np.float32) - petal_color)

    # pixel-wise norm = sqrt(delta_B^2 + delta_G^2 + delta_R^2)
    dist_from_petal_color = np.linalg.norm(diff_from_petal_color,
                                           axis=2)

    # make a temporary 8-bit image to display.
    # need to clip because some distances could be larger than 255 
    tmp = np.clip(dist_from_petal_color, 0, 255).astype(np.uint8)

    show_image(tmp, 'distance from petal color')

    # "masks" are 2D boolean arrays in numpy
    mask = (dist_from_petal_color < 150)

    print(f'mask is {mask.shape}, {mask.dtype}')

    tmp = mask.astype(np.uint8) * 255

    show_image(tmp, 'mask resulting from threshold operation')

    # we can use masks to address groups of pixels within an image

    # make black image
    tmp = np.zeros_like(img)

    # wherever mask != 0, copy from original image
    tmp[mask] = img[mask]

    show_image(tmp, 'just the flower part of the original image')

    # get blurred image copy
    tmp = img_blur_a_lot.copy()

    # wherever mask != 0, copy from original image
    tmp[mask] = img[mask]

    show_image(tmp, 'a neat effect')
    

main()
