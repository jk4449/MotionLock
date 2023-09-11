import numpy as np
import cv2 as cv
import shutil
import os
def largest_contour(c_lst):
    # returns the contour with the largest area from a list of contours.
    l_area = 0
    l_cnt = None
    for c in c_lst:  # go through all contours in the list
        area = cv.contourArea(c)
        if area > l_area:  # if the area of the current contour is the largest encountered
            l_area = area  # set this area as the largest
            l_cnt = c  # set this contour as the largest contour
    return l_cnt  # return the largest contour

def find_handLoc(center_x, center_y, frame_x, frame_y):
    # returns the location of the hand contour
    hand_loc = ""
    if center_y < frame_y / 3:
        hand_loc += "U"
    elif center_y < frame_y * 2 / 3:
        hand_loc += "C"
    else:
        hand_loc += "L"

    if center_x < frame_x / 3:
        hand_loc += "L"
    elif center_x < frame_x * 2 / 3:
        hand_loc += "C"
    else:
        hand_loc += "R"
    accepted = ["UL", "UR", "LR", "LL", "C"]
    if hand_loc == "CC":
        hand_loc = "C"
    elif hand_loc not in accepted:
        hand_loc = "none"
    return hand_loc

def load_contour(ref_img_path, ref_img_name):
    # returns contour from the specified file from the specified folder.

    # load the reference image from the reference image folder
    ref_img = cv.imread(os.path.join(ref_img_path, ref_img_name), 0)

    # find the list of contours from the reference image
    ref_img_contours, _ = cv.findContours(ref_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # find the correct hand contour (largest contour)
    ref_img_contour = largest_contour(ref_img_contours)
    return ref_img_contour

def take_pictures(seq_len, difficulty):
    # take pictures (seq_len) times, convert it to binary image where skin is white and rest is black,
    # and save it in the binary image folder.
    cap = cv.VideoCapture(0)
    count = 0
    while count < seq_len:
        _, frame = cap.read()
        cv.imshow("Capturing", frame)
        key = cv.waitKey(1)
        if key == ord('s'):
            # take a photo when key s is pressed
            # save the image in the Images folder.
            bin_img_name = "Bin_Images/" + difficulty + str(count) + ".jpeg"

            # PROCESSING IMAGE
            # convert image file to HSV
            converted = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            # convert image to binary file that shows skin as white and the rest as black
            skinMask = cv.inRange(converted, skin_lower, skin_upper)
            # save the processed binary image to the folder
            cv.imwrite(bin_img_name, skinMask)
            # increment image count
            count += 1
        elif key == ord('q'):
            # Turn off camera if key q is pressed
            print("Turning off camera.")
            cap.release
            break

if __name__ == "__main__":
    # 1. SET UP / CLEAN UP FOLDERS

    # bin_img_path is the folder annotated binary images would be saved at.
    bin_img_path = '/Users/jungminkim/PycharmProjects/MotionLock/Bin_Images'

    # ref_img_path is where images of a standard fist, palm and splay would be found at.
    # We would compare these standard "reference images" with an unlabeled image to determine its shape.
    ref_img_path = '/Users/jungminkim/PycharmProjects/MotionLock/Ref_Images'

    # clear any images in the folder from previous run
    if os.path.exists(bin_img_path):
        shutil.rmtree(bin_img_path)
    os.mkdir(bin_img_path)

    # 2. SETUP SKIN RANGE AND LOCK SEQUENCE

    # boundaries in color to be considered skin
    '''
    skin range values taken from Adrian Rosebrock's article:
    Rosebrock, Adrian. “Tutorial: Skin Detection Example Using Python and Opencv.” PyImageSearch, 18 Aug. 2014,
    https://pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/.
    skin_upper bound adjusted from [20,255,255] to [20,150,255] to fit my domain.
    '''
    skin_lower = np.array([0, 48, 80], dtype="uint8")
    skin_upper = np.array([20, 150, 255], dtype="uint8")

    # set easy, good and hard sequences
    sequence = {"easy": [("fist", "C"), ("palm", "C"), ("splay", "C")],
                "good": [("fist", "UR"), ("palm", "LR"), ("splay", "UR")],
                "hard": [("splay", "UL"), ("splay", "LL"), ("palm", "UL")]}

    # 3. LOAD CONTOURS FROM REFERENCE IMAGES FOR COMPARISON

    # Three different reference images available for each shape, to capture the slight variation in
    # hand position, hand location, rotation and light.
    ref_imgs = {"palm": ["palm0.jpeg", "palm1.jpeg", "palm2.jpeg"],
                "fist": ["fist0.jpeg", "fist1.jpeg", "fist2.jpeg"],
                "splay": ["splay0.jpeg", "splay1.jpeg", "splay2.jpeg"]}
    # load up the reference images
    palm_ctr = []
    fist_ctr = []
    splay_ctr = []
    for i in range(len(ref_imgs["palm"])):
        palm_ctr.append(load_contour(ref_img_path, ref_imgs["palm"][i]))
    for i in range(len(ref_imgs["fist"])):
        fist_ctr.append(load_contour(ref_img_path, ref_imgs["fist"][i]))
    for i in range(len(ref_imgs["splay"])):
        splay_ctr.append(load_contour(ref_img_path, ref_imgs["splay"][i]))

    # 4. TEST LOCK SEQUENCE
    for mode in sequence:
        print("Show us the " + mode + " lock sequence! Press \"s\" to take a picture")
        # initialize user sequence
        user_sequence = []
        seq_len = len(sequence[mode])
        # take as many pictures as the sequence length and save them in a designated folder.
        take_pictures(seq_len, mode)
        # iterate through the pictures taken and analyze its "what" (hand shape) and "where" (hand location)
        for i in range(seq_len):
            img_name = os.path.join(bin_img_path, mode + str(i) + ".jpeg")
            handFrame = cv.imread(img_name, 0)
            # a. FIND HAND CONTOUR
            img_contours, _ = cv.findContours(handFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # save the shape of the hand in img_contour
            '''
            When I drew the contours out from the list, most contours referred to a very small area 
            near the hand border. Since hand is the biggest shape in the image if my rules on taking 
            the pictures are followed, the shape of the hand would be captured by the largest contour.
            Hence I decided to use area of contour as a metric to select the contour of the hand.
            '''
            img_contour = largest_contour(img_contours)

            # b. DETERMINE "WHERE"
            x, y, w, h = cv.boundingRect(img_contour)
            center_x, center_y = x + w // 2, y + h // 2
            frame_x, frame_y = handFrame.shape[1], handFrame.shape[0]

            # find_handLoc returns where the center of the contour rectangle falls.
            '''
            The image frame is divided into 9 sections: upper left, upper center, upper right, 
            center left, center center, center right, lower left, lower center and lower right.
            find_handLoc determines where the center of the hand contour falls into out of the 9 sections.
            However, the location can be either UL (upper left), LL (lower left), C (center), 
            UR (upper right), or LR (lower right). If the center falls in any other places, 
            it will be labeled as "none".
            '''
            hand_loc = find_handLoc(center_x, center_y, frame_x, frame_y)

            # c. DETERMINE "what"
            '''
            The smaller shape_diff is, the closer the image is to that shape.
            Compare the image with the three reference images of a palm.
            The smallest difference value represents the palm. 
            Continue with other shapes to find the representative value for fist and splay.
            Finally, the shape with the smallest difference value is selected.
            If the smallest difference value is still too big, we identify it as "none".
            '''
            diff_lst = []
            for i in range(3):
                diff_lst.append(cv.matchShapes(img_contour, palm_ctr[i], cv.CONTOURS_MATCH_I1, 0))
            palm_diff = diff_lst[0] # representative value of palm

            diff_lst = []
            for j in range(3):
                diff_lst.append(cv.matchShapes(img_contour, fist_ctr[i], cv.CONTOURS_MATCH_I1, 0))
            fist_diff = diff_lst[0] # representative value of fist

            diff_lst = []
            for k in range(3):
                diff_lst.append(cv.matchShapes(img_contour, splay_ctr[i], cv.CONTOURS_MATCH_I1, 0))
            splay_diff = diff_lst[0] # representative value of splay

            best_match = sorted([("palm", palm_diff), ("fist", fist_diff), ("splay", splay_diff)], key=lambda a: a[1])
            if best_match[0][1] > 0.5:
                # if image is not close to fist, splay or palm, output none.
                handShape = "none"
            else:
                # else, shape of the image is the one with the least difference.
                handShape = best_match[0][0]

            # save the "what" and "where" to the user sequence.
            user_sequence.append((handShape, hand_loc))

            # d. ANNOTATE IMAGE

            # create a frame to annotate with colors.
            colorframe = cv.cvtColor(handFrame, cv.COLOR_GRAY2BGR)

            # annotate1: draw a rectangle surrounding the contour
            cv.rectangle(colorframe, (x, y), (x + w, y + h), (0, 255, 0), 4)
            
            # annotate2: draw vertical boundaries between left, center, right
            cv.line(colorframe, (frame_x//3, 0), (frame_x//3, frame_y), (255, 0, 0), 3)
            cv.line(colorframe, (frame_x*2//3, 0), (frame_x*2//3, frame_y), (255, 0, 0), 3)
            # annotate3: draw horizontal boundaries between upper, center, lower
            cv.line(colorframe, (0, frame_y//3), (frame_x, frame_y//3), (255, 0, 0), 3)
            cv.line(colorframe, (0, frame_y*2//3), (frame_x, frame_y*2//3), (255, 0, 0), 3)
    
            # annotate4: draw a red dot at the center of the contour
            # This explains visually the "where" of the hand gesture.
            cv.circle(colorframe, (center_x, center_y), 2, (0, 0, 255), 2)
            
            # annotate5: type the verdict on the image at the upper left corner
            cv.putText(colorframe, "<" + handShape + ",  " + hand_loc + ">", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2,
                       (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(colorframe, str(best_match), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

            # Save annotated binary intermediate
            cv.imwrite(img_name, colorframe)

        # EVALUATION

        # compare the user's sequence to the lock sequence
        if user_sequence == sequence[mode]:
            # open lock if user got the sequence right.
            print("Lock opened")
        else:
            # print fail message when the user did not get the sequence right.
            print("Wrong sequence")

        # print result for user's analysis.
        print("user_sequence: ", user_sequence)
