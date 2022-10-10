from fileinput import filename
from locale import normalize
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from skimage import io
from scipy.ndimage.filters import gaussian_laplace
def corp_margin(img):
        img2=img.sum(axis=2)
        (row,col)=img2.shape
        row_top=0
        row_down=0
        col_top=0
        col_down=0
        for r in range(0,row):
                if img2.sum(axis=1)[r]<400*col:
                        row_top=r
                        break
 
        for r in range(row-1,0,-1):
                if img2.sum(axis=1)[r]<400*col:
                        row_down=r
                        break
 
        for c in range(0,col):
                if img2.sum(axis=0)[c]<400*row:
                        col_top=c
                        break
 
        for c in range(col-1,0,-1):
                if img2.sum(axis=0)[c]<400*row:
                        col_down=c
                        break
 
        new_img=img[row_top:row_down+1,col_top:col_down+1,0:3]

        return new_img

def divideBRG(filename):
    orig_img= cv2.imread("./data/"+filename+"_cut.jpg",cv2.IMREAD_GRAYSCALE)

    height= int(len(orig_img)/3)
    ##obtain each channel's data
    img_B= orig_img[0:height]
    img_G= orig_img[height:2*height]
    img_R= orig_img[2*height:3*height]
    return img_B,img_G,img_R

def find_offset(c1,c2):  ##c1 is the base channel
    c1 = gaussian_laplace(c1,1)
    c2 = gaussian_laplace(c2,1)
    fft_c1 = np.fft.fft2(c1)
    f1 = np.fft.fftshift(fft_c1)
    fft_c2 = np.fft.fft2(c2)
    f2 = np.fft.fftshift(fft_c2)
    res = f1*np.conjugate(f2)
    inv_res = np.fft.ifft2(res)
    inv_res1=np.fft.fftshift(inv_res)
    i,j = np.unravel_index(inv_res.argmax(), inv_res.shape)
    h,w = inv_res.shape
    print(h,w)
    normalized_ifft = np.zeros((h,w))
    cv2.normalize(np.abs(inv_res1), normalized_ifft, 0, 255, cv2.NORM_MINMAX)
    return i,j,normalized_ifft

def find_offset_nopre(c1,c2):  ##c1 is the base channel
    fft_c1 = np.fft.fft2(c1)
    f1 = np.fft.fftshift(fft_c1)
    fft_c2 = np.fft.fft2(c2)
    f2 = np.fft.fftshift(fft_c2)
    res = f1*np.conjugate(f2)
    inv_res = np.fft.ifft2(res)
    inv_res1=np.fft.fftshift(inv_res)

    i,j = np.unravel_index(inv_res.argmax(), inv_res.shape)
    
    h,w = inv_res.shape
    normalized_ifft = np.zeros((h,w))
    cv2.normalize(np.abs(inv_res1), normalized_ifft, 0, 255, cv2.NORM_MINMAX)
    return i,j,normalized_ifft




if __name__=="__main__":
    filenames= ["00125v","00149v","00153v","00351v","00398v","01112v"]
    for filename in filenames:
        B,G,R = divideBRG(filename)
        iG,jG,normalized_ifft_G= find_offset(B,G)
        print("The offset for align G channel for "+filename+ "is", (iG,jG))
        iR,jR, normalized_ifft_R= find_offset(B,R)
        print("The offset for align R channel for "+filename+ "is", (iR,jR))
        shift_G = np.roll(G,(iG,jG),(0,1))
        shift_R = np.roll(R,(iR,jR),(0,1))
        res = cv2.merge((B,shift_G,shift_R))
        cv2.imwrite("./output/result/"+filename+".jpg", res)
        cv2.imwrite("./output/ifft/"+filename+"_G.jpg", normalized_ifft_G)
        cv2.imwrite("./output/ifft/"+filename+"_R.jpg", normalized_ifft_R)

    # filenames= ["00125v","00149v","00153v","00351v","00398v","01112v"]
    # for filename in filenames:
    #     img = cv2.imread("./data/"+filename+".jpg", cv2.IMREAD_COLOR)
    #     img_c = corp_margin(img)
    #     cv2.imwrite("./data/"+filename+"_cut.jpg", img_c)

        iG1,jG1,normalized_ifft_G1= find_offset_nopre(B,G)
        iR1,jR1, normalized_ifft_R1= find_offset_nopre(B,R)
        cv2.imwrite("./output/ifft_nopre/"+filename+"_Gn.jpg", normalized_ifft_G1)
        cv2.imwrite("./output/ifft_nopre/"+filename+"_Rn.jpg", normalized_ifft_R1)

    filenames_hires= ["01047u","01657u","01861a"]
    for filename in filenames_hires:
        B,G,R = divideBRG(filename)
        start1=time.time()
        iG,jG,normalized_ifft_G= find_offset(B,G)
        end1= time.time()
        print("Time for find G channel offset for"+filename+".tif is", end1-start1)
        print("The offset for align G channel for "+filename+ "_tif is", (iG,jG))
        
        start2=time.time()
        iR,jR, normalized_ifft_R= find_offset(B,R)
        end2= time.time()
        print("Time for find R channel offset for"+filename+".tif is", end2-start2)
        print("The offset for align R channel for "+filename+ "_tif is", (iR,jR))

        shift_G = np.roll(G,(iG,jG),(0,1))
        shift_R = np.roll(R,(iR,jR),(0,1))
        res = cv2.merge((B,shift_G,shift_R))
        cv2.imwrite("./output/result/"+filename+"tif.jpg", res)
        cv2.imwrite("./output/ifft/"+filename+"_G_tif.jpg", normalized_ifft_G)
        cv2.imwrite("./output/ifft/"+filename+"_R_tif.jpg", normalized_ifft_R)

        iG1,jG1,normalized_ifft_G1= find_offset_nopre(B,G)
        iR1,jR1, normalized_ifft_R1= find_offset_nopre(B,R)
        cv2.imwrite("./output/ifft_nopre/"+filename+"_Gn_tif.jpg", normalized_ifft_G1)
        cv2.imwrite("./output/ifft_nopre/"+filename+"_Rn_tif.jpg", normalized_ifft_R1)
    
    
    
    # filenames= ["01047u","01657u","01861a"]
    # for filename in filenames:
    #     img = cv2.imread("./data/"+filename+".tif", cv2.IMREAD_COLOR)
    #     img_c = corp_margin(img)
    #     cv2.imwrite("./data/"+filename+"_cut.jpg", img_c)
    
    
    
    
    
    
    
    

