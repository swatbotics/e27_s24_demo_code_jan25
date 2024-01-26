import cv2
import sys

# for downscaling
def size_at_most(shape, max_height):
    w, h = shape
    scl = 1
    while h > max_height:
        w, h = w//2, h//2
        scl *= 2
    return (w, h), scl

# our main function
def main():

    # Load our cascade classifier
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # You can specify a video device (e.g. 0) on the command line, a
    # movie file, or a static image filename
    if len(sys.argv) != 2:
        print('usage: python anonymize2.py DEVICENUM')
        print('  e.g. python anonymize2.py 0')
        sys.exit(1)
        
    device_num = int(sys.argv[1])
    cap = cv2.VideoCapture(device_num)

    # Create our window
    cv2.namedWindow('win')

    # Hit the 'b' key to toggle blurring
    do_blur = False

    detect_size = None
    tiny_size = None

    # Main loop
    while True:

        # Pull an image from capture
        ok, img = cap.read()
        if not ok or img is None:
            break

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Downsample large images to feed to detector - important for runtime
        if detect_size is None:
            orig_size = img_gray.shape[1], img_gray.shape[0]
            detect_size, detect_scl = size_at_most(orig_size, 400)
            tiny_size, _ = size_at_most(orig_size, 40)

        img_small = cv2.resize(img_gray, detect_size, 
                               interpolation=cv2.INTER_AREA)

        # Run the cascade classifier
        rects = cascade.detectMultiScale(img_small)

        # Really downscale and then upscale with nearest neighbor to add mosaic blur
        img_tiny = cv2.resize(img, tiny_size, interpolation=cv2.INTER_AREA)
        img_blurry = cv2.resize(img_tiny, orig_size, interpolation=cv2.INTER_NEAREST)

        # Visualize the returned rectangles
        for rect in rects:
            x, y, w, h = rect * detect_scl
            if do_blur:
                img[y:y+h, x:x+w] = img_blurry[y:y+h, x:x+w]
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

        # Show it
        cv2.imshow('win', img)

        # Handle key presses - 'b' toggles blur
        k = cv2.waitKey(5)
        if k == 27:
            break
        elif k == ord('b'):
            do_blur = not do_blur

if __name__ == '__main__':
    main()
