
import cv2

def crop_sample(image, saveid=None, savedir='../data/straight/'):
    """
    functionality: Take an input image, detects faces, save each detected faces
    inputs:     image: input image: dir + filename
    output:     save grayscale version of detected (face + hair) sample in .jpg format
    remark: credit to http://gregblogs.com/computer-vision-cropping-faces-from-images-using-opencv2/
    """
    facedata = "haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image, 0)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))# uncomment to draw rectangle
        # hueristic estimate of increasing cropping area around detected face to include hair
        xmargin = int(w/6)
        ymargin = int(h/1.9)
        ystart = (y-ymargin) if (y-ymargin)>0 else 0
        yend = (y+h+ymargin) if (y+h+ymargin)<img.shape[1] else img.shape[1]
        xstart = (x-xmargin) if (x-xmargin)>0 else 0
        xend = (x+w+xmargin) if (x+w+xmargin)<img.shape[0] else img.shape[0]
        sub_face = img[ystart:yend, xstart:xend]
        if saveid==None:
            import random
            saveid = random.randint(0, 100)
        face_file_name = savedir+"sample_" + str(saveid) + ".jpg"
        cv2.imwrite(face_file_name, sub_face)
        saveid = saveid+1
    return

def test_crop_sample():
    '''test crop_sample func - navigate to savedir to view cropped samples'''
    sample_name = "/Users/Zhongyu/Documents/projects/insight/UnsuperGAN/straight_hair/straight_hair3/pic_010.jpg"
    crop_sample(sample_name, saveid=0, savedir="./")

if __name__ == '__main__':  
    test_crop_sample()

    