import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def divideBRG(filename):
    orig_img= cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    height= int(len(orig_img)/3)
    ##obtain each channel's data
    img_B= orig_img[0:height]

    img_G= orig_img[height:2*height]

    img_R= orig_img[2*height:3*height]
    
    ###use this code for ssd mehod for mul_alignment
    # img_B= orig_img[0:height]/255

    # img_G= orig_img[height:2*height]/255

    # img_R= orig_img[2*height:3*height]/255

    return img_B,img_G,img_R


def SSD(shift_image,fix_image):
    return ((shift_image-fix_image)**2).sum()

def NCC(image1,image2):
    return cv2.matchTemplate(np.float32(image1 - image1.mean()),np.float32(image2 - image2.mean()),cv2.TM_CCORR_NORMED)


def basic_align(window,image_shift,image_fix):

    # pass
    score_ssd= math.inf
    score_ncc= -math.inf
    offset_ssd=(0,0)
    offset_ncc=(0,0)
     ##first is the ssd method, second is the ncc method.
    for x in range(-window,window+1):
        for y in range(-window,window+1):
            image_roll = np.roll(image_shift,(x,y),(1,0))
            score1=SSD(image_roll,image_fix)
            if score1<score_ssd:
                score_ssd= score1
                offset_ssd= (x,y)

            score2= NCC(image_roll,image_fix)
            if score2>score_ncc:
                score_ncc=score2
                offset_ncc=(x,y)
    print(offset_ssd,offset_ncc)
    return offset_ssd,offset_ncc

def mul_align(window,image_shift,image_fix):
    ###define image shape larger than 10 times of the window size
    score_ncc= -math.inf
    score_ssd= math.inf
    if image_shift.shape[0]> (window*10) and image_shift.shape[1]>(window*10):
        down_sample_image_shift= cv2.pyrDown(image_shift)
        down_sample_image_fix= cv2.pyrDown(image_fix)
        dis1,dis2=mul_align(window,down_sample_image_shift,down_sample_image_fix)
        dis1=(dis1[0]*2,dis1[1]*2)
        dis2=(dis2[0]*2,dis2[1]*2)
    else:
        dis1= (0,0)
        dis2= (0,0)
        
    
    for x in range(-window,window+1):
        for y in range(-window,window+1):
            image_roll_ssd = np.roll(image_shift,(dis1[0]+x,dis1[1]+y),(1,0))
            image_roll_ncc= np.roll(image_shift,(dis2[0]+x,dis2[1]+y),(1,0))
            score1=SSD(image_roll_ssd,image_fix)
            if score1<score_ssd:
                score_ssd= score1
                offset1= (x,y)

            score2= NCC(image_roll_ncc,image_fix)
            if score2 >score_ncc:
                score_ncc=score2
                offset2=(x,y)
    dis1=(dis1[0]+offset1[0],dis1[1]+offset1[1])
    dis2=(dis2[0]+offset2[0],dis2[1]+offset2[1])

    print(dis1,dis2)
    return dis1,dis2



def merge_image(window,image_fix,B,G,R):
    if image_fix=="B":
    ##B here is the fixed channel    
        offset_BG_ssd, offset_BG_ncc = basic_align(window,G,B)
        offset_BR_ssd, offset_BR_ncc = basic_align(window,R,B)
        ##ssd method
        shift_G_ssd= np.roll(G,offset_BG_ssd,(1,0))
        shift_R_ssd= np.roll(R,offset_BR_ssd,(1,0))
        res1=cv2.merge((B,shift_G_ssd,shift_R_ssd))
       
        ##ncc method 
        shift_G_ncc= np.roll(G,offset_BG_ncc,(1,0))
        shift_R_ncc= np.roll(R,offset_BR_ncc,(1,0))
        res2=cv2.merge((B,shift_G_ncc,shift_R_ncc))
        
    
    if image_fix=="G":
    ##G here is the fixed channel   
        offset_GB_ssd, offset_GB_ncc = basic_align(window,B,G)
        offset_GR_ssd, offset_GR_ncc = basic_align(window,R,G)
        ##ssd method
        shift_B_ssd= np.roll(B,offset_GB_ssd,(1,0))
        shift_R_ssd= np.roll(R,offset_GR_ssd,(1,0))
        res1=cv2.merge((shift_B_ssd,G,shift_R_ssd))

        ##ncc method 
        shift_B_ncc= np.roll(G,offset_GB_ncc,(1,0))
        shift_R_ncc= np.roll(R,offset_GR_ncc,(1,0))
        res2=cv2.merge((shift_B_ncc,G,shift_R_ncc))
        

    if image_fix=="R":
    ##R here is the fixed channel   
        offset_RB_ssd, offset_RB_ncc = basic_align(window,B,R)
        offset_RG_ssd, offset_RG_ncc = basic_align(window,G,R)
        ##ssd method
        shift_B_ssd= np.roll(B,offset_RB_ssd,(1,0))
        shift_G_ssd= np.roll(R,offset_RG_ssd,(1,0))
        res1=cv2.merge((shift_B_ssd,shift_G_ssd,R))

        ##ncc method 
        shift_B_ncc= np.roll(G,offset_RB_ncc,(1,0))
        shift_G_ncc= np.roll(R,offset_RG_ncc,(1,0))
        res2=cv2.merge((shift_B_ncc,shift_G_ncc,R))
        
    return res1,res2

def merge_image_hires(window,image_fix,B,G,R):
    if image_fix=="B":
    ##B here is the fixed channel    
        offset_BG_ssd, offset_BG_ncc = mul_align(window,G,B)
        offset_BR_ssd, offset_BR_ncc = mul_align(window,R,B)
        ##ssd method
        shift_G_ssd= np.roll(G,offset_BG_ssd,(1,0))
        shift_R_ssd= np.roll(R,offset_BR_ssd,(1,0))
        res1=cv2.merge((B,shift_G_ssd,shift_R_ssd))
       
        ##ncc method 
        shift_G_ncc= np.roll(G,offset_BG_ncc,(1,0))
        shift_R_ncc= np.roll(R,offset_BR_ncc,(1,0))
        res2=cv2.merge((B,shift_G_ncc,shift_R_ncc))
        
    
    if image_fix=="G":
    ##G here is the fixed channel   
        offset_GB_ssd, offset_GB_ncc = mul_align(window,B,G)
        offset_GR_ssd, offset_GR_ncc = mul_align(window,R,G)
        ##ssd method
        shift_B_ssd= np.roll(B,offset_GB_ssd,(1,0))
        shift_R_ssd= np.roll(R,offset_GR_ssd,(1,0))
        res1=cv2.merge((shift_B_ssd,G,shift_R_ssd))

        ##ncc method 
        shift_B_ncc= np.roll(G,offset_GB_ncc,(1,0))
        shift_R_ncc= np.roll(R,offset_GR_ncc,(1,0))
        res2=cv2.merge((shift_B_ncc,G,shift_R_ncc))
        

    if image_fix=="R":
    ##R here is the fixed channel   
        offset_RB_ssd, offset_RB_ncc = mul_align(window,B,R)
        offset_RG_ssd, offset_RG_ncc = mul_align(window,G,R)
        ##ssd method
        shift_B_ssd= np.roll(B,offset_RB_ssd,(1,0))
        shift_G_ssd= np.roll(R,offset_RG_ssd,(1,0))
        res1=cv2.merge((shift_B_ssd,shift_G_ssd,R))

        ##ncc method 
        shift_B_ncc= np.roll(G,offset_RB_ncc,(1,0))
        shift_G_ncc= np.roll(R,offset_RG_ncc,(1,0))
        res2=cv2.merge((shift_B_ncc,shift_G_ncc,R))
        
    return res1,res2

    

#####basic alignment
# filenames=["00125v", "00149v", "00153v", "00351v", "00398v", "01112v"]
bases = ["B","G","R"]
# for filename in filenames:
#     path="data/"+filename+".jpg"
#     B,G,R=divideBRG(path)
#     window=10

#     for base in bases:

#         res1,res2=merge_image(window,base,255*B,255*G,255*R)             
#         cv2.imwrite('./result/'+filename+ '_ssd_'+base+'.jpg',res1)
#         cv2.imwrite('./result/'+filename+ '_ncc_'+base+'.jpg',res2)


###mul alignment
filenames = [ "01047u", "01657u", "01861a"]
for filename in filenames:
    path="data_hires/"+filename+".tif"
    B,G,R=divideBRG(path)
    window=10
    for base in bases:
        start_time = time.time()
        res1,res2=merge_image_hires(window,base,B,G,R) 
        
        ###use this for ssd method for mul_alignment
        # res1,res2=merge_image_hires(window,base,255*B,255*G,255*R)         
        cv2.imwrite('./result_hires/'+filename+ '_ssd_'+base+'.jpg',res1)
        cv2.imwrite('./result_hires/'+filename+ '_ncc_'+base+'.jpg',res2)
        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)

    
    
    
    
    
    
    
    
    
    
    
    
    

