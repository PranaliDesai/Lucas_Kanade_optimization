import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
cv_img = []
path = '/home/pranali/Desktop/673/p4/data/vase/'

for image in os.listdir(path):
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    cv_img.append(img)

    
def LucasKanadefunction(initialtemp, initialtemp1, rectpoints, pos0=np.zeros(2)):
    threshold = 0.0001
    x1, y1, x2, y2, x3, y3, x4, y4 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3], rectpoints[4], rectpoints[5], rectpoints[6], rectpoints[7]
    initial_y, initial_x = np.gradient(initialtemp1)
    dp = 1

    while np.square(dp).sum() > threshold:
        # warp image
        posx, posy = pos0[0], pos0[1]
        x1_warp, y1_warp, x2_warp, y2_warp, x3_warp, y3_warp, x4_warp, y4_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy, x3 + posx, y3 + posy, x4 + posx, y4 + posy

        x = np.arange(0, initialtemp.shape[0], 1)
        y = np.arange(0, initialtemp.shape[1], 1)

        a1 = np.linspace(x1, x3, 87)
        b1 = np.linspace(y1, y3, 36)
        a2 = np.linspace(x4, x2, 87)
        b2 = np.linspace(y2, y4, 36)
        a = np.union1d(a1, a2)
        b = np.union1d(b1, b2)
        aa, bb = np.meshgrid(a, b)

        a1_warp = np.linspace(x1_warp, x3_warp, 87)
        b1_warp = np.linspace(y1_warp, y3_warp, 36)
        a2_warp = np.linspace(x4_warp, x2_warp, 87)
        b2_warp = np.linspace(y2_warp, y4_warp, 36)
        a_warp = np.union1d(a1_warp, a2_warp)
        b_warp = np.union1d(b1_warp, b2_warp)
        aaw, bbw = np.meshgrid(a_warp, b_warp)

        spline = RectBivariateSpline(x, y, initialtemp)
        T = spline.ev(bb, aa)

        spline1 = RectBivariateSpline(x, y, initialtemp1)
        warpImg = spline1.ev(bbw, aaw)

        # compute error image
        error = T - warpImg
        errorImg = error.reshape(-1, 1)

        # compute gradient
        spline_gx = RectBivariateSpline(x, y, initial_x)
        initial_x_w = spline_gx.ev(bbw, aaw)

        spline_gy = RectBivariateSpline(x, y, initial_y)
        initial_y_w = spline_gy.ev(bbw, aaw)
        # I is (n,2)
        I = np.vstack((initial_x_w.ravel(), initial_y_w.ravel())).T
        #print(I.shape)
        # evaluate jacobian (2,2)
        jacobian = np.array([[1, 0], [0, 1]])

        # computer Hessian
        hessian = I @ jacobian
        # H is (2,2)
        H = hessian.T @ hessian

        # compute dp
        # dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (hessian.T) @ errorImg

        # update parameters
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p

rectpoints1 = [124,91,172,91,172,150,124,150]
rectpoints2 = [62, 48, 85, 48, 85, 74, 62, 74]
rectpoints3 = [31, 22, 43, 22, 43, 37, 31, 37]
rectpoints4 = [16, 13, 21, 13, 21, 18, 16, 18]
rectpoints10 = copy.deepcopy(rectpoints1)
rectpoints20 = copy.deepcopy(rectpoints2)
rectpoints30 = copy.deepcopy(rectpoints3)
rectpoints40 = copy.deepcopy(rectpoints4)

cap_140 = cv_img[0]
cap_140 = cv2.GaussianBlur(cap_140, (9,9), 0)
cap_gray_140_1 = cv2.cvtColor(cap_140, cv2.COLOR_BGR2GRAY)
cap_gray_140_2 = cv2.pyrDown(cap_gray_140_1)
cap_gray_140_3 = cv2.pyrDown(cap_gray_140_2)
#print('shape', cap_140.shape)
for i in range(0, len(cv_img)-1):
    #print(i)
    image_index = i
    cap = cv_img[image_index]
    cap_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    #cv2.rectangle(cap,(int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),3)
    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[2]), int(rectpoints1[3])), (0,255,0), 3)
    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[6]), int(rectpoints1[7])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[2]), int(rectpoints1[3])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[6]), int(rectpoints1[7])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)

    cv2.imshow('Vase', cap)
    cap_next = cv_img[image_index+1]
    cap_next = cv2.GaussianBlur(cap_next, (9,9), 0)
    cap_gray_next1 = cv2.cvtColor(cap_next, cv2.COLOR_BGR2GRAY)
    cap_gray_next2 = cv2.pyrDown(cap_gray_next1)
    cap_gray_next3 = cv2.pyrDown(cap_gray_next2)
    
    initialtemp0 = cap_gray_140_3 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next3 / 255.
    p1 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints30)
    
    initialtemp0 = cap_gray_140_2 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next2 / 255.
    p2 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints20, pos0 = np.array(p1)*2)
    
    initialtemp0 = cap_gray_140_1 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next1 / 255.
    p3 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints10, pos0 = np.array(p1)*4 + np.array(p2)*2)
    
    rectpoints1[0] = rectpoints10[0] + p3[0]
    rectpoints1[1] = rectpoints10[1] + p3[1]
    
    rectpoints1[2] = rectpoints10[2] + p3[0]
    rectpoints1[3] = rectpoints10[3] + p3[1]
    
    rectpoints1[4] = rectpoints10[4] + p3[0]
    rectpoints1[5] = rectpoints10[5] + p3[1]
    
    rectpoints1[6] = rectpoints10[6] + p3[0]
    rectpoints1[7] = rectpoints10[7] + p3[1]
    
    if (image_index > 16 and image_index < 30) or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[1] = 90
        rectpoints1[3] = 90
        
    if image_index > 16 and image_index< 30 or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[0] = rectpoints1[0] - 20
        rectpoints1[6] = rectpoints1[6] - 20
        
    #print(rectpoints1)
    #break
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
cv_img = []
path = '/home/pranali/Desktop/673/p4/data/vase/'

for image in os.listdir(path):
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    cv_img.append(img)

    
def LucasKanadefunction(initialtemp, initialtemp1, rectpoints, pos0=np.zeros(2)):
    threshold = 0.0001
    x1, y1, x2, y2, x3, y3, x4, y4 = rectpoints[0], rectpoints[1], rectpoints[2], rectpoints[3], rectpoints[4], rectpoints[5], rectpoints[6], rectpoints[7]
    initial_y, initial_x = np.gradient(initialtemp1)
    dp = 1

    while np.square(dp).sum() > threshold:
        # warp image
        posx, posy = pos0[0], pos0[1]
        x1_warp, y1_warp, x2_warp, y2_warp, x3_warp, y3_warp, x4_warp, y4_warp = x1 + posx, y1 + posy, x2 + posx, y2 + posy, x3 + posx, y3 + posy, x4 + posx, y4 + posy

        x = np.arange(0, initialtemp.shape[0], 1)
        y = np.arange(0, initialtemp.shape[1], 1)

        a1 = np.linspace(x1, x3, 87)
        b1 = np.linspace(y1, y3, 36)
        a2 = np.linspace(x4, x2, 87)
        b2 = np.linspace(y2, y4, 36)
        a = np.union1d(a1, a2)
        b = np.union1d(b1, b2)
        aa, bb = np.meshgrid(a, b)

        a1_warp = np.linspace(x1_warp, x3_warp, 87)
        b1_warp = np.linspace(y1_warp, y3_warp, 36)
        a2_warp = np.linspace(x4_warp, x2_warp, 87)
        b2_warp = np.linspace(y2_warp, y4_warp, 36)
        a_warp = np.union1d(a1_warp, a2_warp)
        b_warp = np.union1d(b1_warp, b2_warp)
        aaw, bbw = np.meshgrid(a_warp, b_warp)

        spline = RectBivariateSpline(x, y, initialtemp)
        T = spline.ev(bb, aa)

        spline1 = RectBivariateSpline(x, y, initialtemp1)
        warpImg = spline1.ev(bbw, aaw)

        # compute error image
        error = T - warpImg
        errorImg = error.reshape(-1, 1)

        # compute gradient
        spline_gx = RectBivariateSpline(x, y, initial_x)
        initial_x_w = spline_gx.ev(bbw, aaw)

        spline_gy = RectBivariateSpline(x, y, initial_y)
        initial_y_w = spline_gy.ev(bbw, aaw)
        # I is (n,2)
        I = np.vstack((initial_x_w.ravel(), initial_y_w.ravel())).T
        #print(I.shape)
        # evaluate jacobian (2,2)
        jacobian = np.array([[1, 0], [0, 1]])

        # computer Hessian
        hessian = I @ jacobian
        # H is (2,2)
        H = hessian.T @ hessian

        # compute dp
        # dp is (2,2)@(2,n)@(n,1) = (2,1)
        dp = np.linalg.inv(H) @ (hessian.T) @ errorImg

        # update parameters
        pos0[0] += dp[0, 0]
        pos0[1] += dp[1, 0]

    p = pos0
    return p

rectpoints1 = [124,91,172,91,172,150,124,150]
rectpoints2 = [62, 48, 85, 48, 85, 74, 62, 74]
rectpoints3 = [31, 22, 43, 22, 43, 37, 31, 37]
rectpoints4 = [16, 13, 21, 13, 21, 18, 16, 18]
rectpoints10 = copy.deepcopy(rectpoints1)
rectpoints20 = copy.deepcopy(rectpoints2)
rectpoints30 = copy.deepcopy(rectpoints3)
rectpoints40 = copy.deepcopy(rectpoints4)

cap_140 = cv_img[0]
cap_140 = cv2.GaussianBlur(cap_140, (9,9), 0)
cap_gray_140_1 = cv2.cvtColor(cap_140, cv2.COLOR_BGR2GRAY)
cap_gray_140_2 = cv2.pyrDown(cap_gray_140_1)
cap_gray_140_3 = cv2.pyrDown(cap_gray_140_2)
#print('shape', cap_140.shape)
for i in range(0, len(cv_img)-1):
    #print(i)
    image_index = i
    cap = cv_img[image_index]
    cap_gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    #cv2.rectangle(cap,(int(rectpoints[0]),int(rectpoints[1])),(int(rectpoints[0])+length,int(rectpoints[1])+width),(0,255,0),3)
    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[2]), int(rectpoints1[3])), (0,255,0), 3)
    cv2.line(cap, (int(rectpoints1[0]), int(rectpoints1[1])), (int(rectpoints1[6]), int(rectpoints1[7])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[2]), int(rectpoints1[3])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)
    cv2.line(cap, (int(rectpoints1[6]), int(rectpoints1[7])), (int(rectpoints1[4]), int(rectpoints1[5])), (0, 255, 0), 3)

    cv2.imshow('Vase', cap)
    cap_next = cv_img[image_index+1]
    cap_next = cv2.GaussianBlur(cap_next, (9,9), 0)
    cap_gray_next1 = cv2.cvtColor(cap_next, cv2.COLOR_BGR2GRAY)
    cap_gray_next2 = cv2.pyrDown(cap_gray_next1)
    cap_gray_next3 = cv2.pyrDown(cap_gray_next2)
    
    initialtemp0 = cap_gray_140_3 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next3 / 255.
    p1 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints30)
    
    initialtemp0 = cap_gray_140_2 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next2 / 255.
    p2 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints20, pos0 = np.array(p1)*2)
    
    initialtemp0 = cap_gray_140_1 / 255.
    #initialtemp = cap_gray / 255.
    initialtemp1 = cap_gray_next1 / 255.
    p3 = LucasKanadefunction(initialtemp0, initialtemp1, rectpoints10, pos0 = np.array(p1)*4 + np.array(p2)*2)
    
    rectpoints1[0] = rectpoints10[0] + p3[0]
    rectpoints1[1] = rectpoints10[1] + p3[1]
    
    rectpoints1[2] = rectpoints10[2] + p3[0]
    rectpoints1[3] = rectpoints10[3] + p3[1]
    
    rectpoints1[4] = rectpoints10[4] + p3[0]
    rectpoints1[5] = rectpoints10[5] + p3[1]
    
    rectpoints1[6] = rectpoints10[6] + p3[0]
    rectpoints1[7] = rectpoints10[7] + p3[1]
    
    if (image_index > 16 and image_index < 30) or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[1] = 90
        rectpoints1[3] = 90
        
    if image_index > 16 and image_index< 30 or (image_index > 56 and image_index < 68) or (image_index > 81 and image_index < 99):
        rectpoints1[0] = rectpoints1[0] - 20
        rectpoints1[6] = rectpoints1[6] - 20
        
    #print(rectpoints1)
    #break
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

