
import cv2
from os import listdir, path

def get_dirs(parent_dir):
	"""
	functionality: Get only directories (dir name not starts with "."") live in parent_dir
	input: 		parent_dir: the parent directory to search in
	output: 	child_dirs: list of child directory names only
	"""
	child_dirs = [fn for fn in listdir(parent_dir) if not fn.startswith('.') \
				 and not path.isfile(path.join(parent_dir, fn))]
	return child_dirs

def get_fns(parent_dir):
	"""
	functionality: Get only filenames (file name not starts with ".")
	input: 		parent_dir: the parent directory to search in
	output: 	filenames: list of filenames only
	"""
	filenames = [fn for fn in listdir(parent_dir) if not fn.startswith('.') \
				and path.isfile(path.join(parent_dir, fn))]
	return filenames

def construct_imgfn(parent_dir):
	"""
	functionality: Get image filenames stored under input directory
	input: 		parent_dir: parent directory where images live under
	output: 	all_imgfns: list of image filenames
	"""
	child_dirs = get_dirs(parent_dir)
	#img_fns = get_fns(parent_dir)
	all_imgfns = []
	for child_dir in child_dirs:
		img_dir = path.join(parent_dir, child_dir)
		img_fns = get_fns(img_dir)
		img_dirfn = [path.join(img_dir, fn) for fn in img_fns]
		all_imgfns += img_dirfn
	return all_imgfns

def batch_crop(parent_dir, savedir):
	"""
	functionality: Batch read in scrapped images, perform face detection and cropping and saving images
	input: 		parent_dir: parent directory where scrapped images are stored
				savedir: locations for saving cropped images
	output: 	None. (cropped images saved in savedir)
	"""
	saveid = 0
	all_imgfns = construct_imgfn(parent_dir)
	for img in all_imgfns:
		saveid = crop_sample(img, savedir, saveid)
		if saveid % 100 ==0:
			print(("saved No. %d image sample...") %(saveid))
	print(("finished batch cropping scrapped images! Saved %d cropped images.") %(saveid))
	return saveid


def run_batch_crop():
	'''Run batch_crop - navigate to savedir to view saved cropped images'''
	parent_dir = "/Users/Zhongyu/Documents/projects/insight/UnsuperGAN/straight_hair/"
	savedir = '/Users/Zhongyu/Documents/projects/insight/UnsuperGAN/data/crop/straight/'
	saveid = batch_crop(parent_dir, savedir)


def crop_sample(image, savedir, saveid=None):
    """
    functionality: Take an input image, detects faces, save each detected faces
    inputs:     image: input image: dir + filename
    output:     updated saveid (and saving grayscale of detected (face + hair) sample in .jpg format)
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
    return saveid

def test_crop_sample():
    '''test crop_sample func - navigate to savedir to view cropped samples'''
    sample_name = "/Users/Zhongyu/Documents/projects/insight/UnsuperGAN/straight_hair/straight_hair3/pic_012.jpg"
    crop_sample(sample_name,  savedir="./", saveid=1)



if __name__ == '__main__':  
    test_crop_sample()
    run_batch_crop()

    