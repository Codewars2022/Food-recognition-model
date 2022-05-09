# organize imports
import cv2
import imutils
import numpy as np
import keras


categories = {
    0: 'Apple',
    1: 'Banana',
    2: 'Cherry',
    3: 'Lemon',
    4: 'Pineapple',
    5: 'Tomato'
}

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


#---------------------------------------------
# To segment the region of food in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the food
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def loadModel():
    model = keras.models.load_model("G:/UMKC/Python Project/Food_detection_calorie_estimation.h5")
    return model

def predict(predict_image, model):
    IMG_SIZE = 120
    ia = cv2.imread(predict_image,cv2.IMREAD_GRAYSCALE)
    ia = cv2.resize(ia, (IMG_SIZE, IMG_SIZE))
    predict = np.array(ia).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predict = predict.astype('float32')
    predict /= 255

    return np.argmax(model.predict(predict))


def capture_image():
    # Loading the Model
    model = loadModel()

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the food region
            food = segment(gray,25)

            # check whether food region is segmented
            if food is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = food

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                
                if(num_frames%100 == 0):
                
                    print("food type")
                    print(type(thresholded))
                    print(thresholded.shape)
                    print(type(food))

                    print("capturing the food region")
                    filename1 = 'capturedImageAtFrame-' + str(num_frames) + ".jpg"
                    print(type(clone))
                    cv2.imwrite(filename1,thresholded)
                    print("predicting" + str(num_frames))
                    predicted = predict(filename1,model)
                    print(str(predicted))
                    print(str(predicted) + "--" + categories[predicted])

        # draw the segmented food
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented food
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            # free up memory
            camera.release()
            cv2.destroyAllWindows()
    return


capture_image()