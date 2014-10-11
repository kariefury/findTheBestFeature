import numpy as np
import wolframalpha
import cv2
import urllib
import numpy as np


from matplotlib import pyplot as plt

def orbFeature(myJPG):
    # img = cv2.imread(myJPG,0)
    req = urllib.urlopen(myJPG)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr,0) # 'load it as it is' (-1 is color?)

    # Initiate STAR detector
    orb = cv2.ORB()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()
    
if __name__ == "__main__":
    print 'here I am'
    client = wolframalpha.Client("WJYTHW-WVLY98YW77")
    res = client.query('define universe')
    resString = next(res.results).text
    print resString
    orbFeature("http://web.eecs.umich.edu/~jiadeng/img/imagenet_icon.jpg")

print 'all done now'