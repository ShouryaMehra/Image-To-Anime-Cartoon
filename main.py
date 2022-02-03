import argparse
import time
import numpy as np
import numpy
from collections import defaultdict
from scipy import stats
import cv2
from PIL import Image, ImageDraw
from flask import Flask,jsonify,request,send_file
import json
import io
from dotenv import load_dotenv
import os

def cartoonize(image):
    """
    convert image into cartoon-like image
    image: input PIL image
    """

    output = np.array(image)
    x, y, c = output.shape
    # hists = []
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)
        # hist, _ = np.histogram(output[:, :, i], bins=np.arange(256+1))
        # hists.append(hist)
    edge = cv2.Canny(output, 100, 200)

    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []
    #H
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180+1))
    hists.append(hist)
    #S
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256+1))
    hists.append(hist)
    #V
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256+1))
    hists.append(hist)

    C = []
    for h in hists:
        C.append(k_histogram(h))
    print("centroids: {0}".format(C))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    contours, _ = cv2.findContours(edge,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     tmp = contours[i]
    #     contours[i] = cv2.approxPolyDP(tmp, 2, False)
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    return output

def update_C(C, hist):
    """
    update centroids until they don't change
    """
    while True:
        groups = defaultdict(list)
        #assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C-i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))
        if np.sum(new_C-C) == 0:
            break
        C = new_C
    return C, groups

def k_histogram(hist):
    """
    choose the best K for k-means and get the centroids
    """
    alpha = 0.001              # p-value threshold for normaltest
    N = 80                      # minimun group size for normaltest
    C = np.array([128])

    while True:
        C, groups = update_C(C, hist)

        #start increase K if possible
        new_C = set()     # use set to avoid same value when seperating centroid
        for i, indice in groups.items():
            #if there are not enough values in the group, do not seperate
            if len(indice) < N:
                new_C.add(C[i])
                continue

            # judge whether we should seperate the centroid
            # by testing if the values of the group is under a
            # normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                #not a normal dist, seperate
                left = 0 if i == 0 else C[i-1]
                right = len(hist)-1 if i == len(C)-1 else C[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (C[i]+left)/2
                    c2 = (C[i]+right)/2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    # though it is not a normal dist, we have no
                    # extra space to seperate
                    new_C.add(C[i])
            else:
                # normal dist, no need to seperate
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

# print(secret_id)
def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message


@app.route('/Cartoon_maker',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        try:
            # convert image to cartoon
            img_params =request.files['cartoon_image'].read()
            npimg = numpy.fromstring(img_params, numpy.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            output = cartoonize(image)
        except:
            # convert image to sketch
            img_params =request.files['sketch_image'].read()
            npimg = numpy.fromstring(img_params, numpy.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # load image

            scale_percent = 0.5 # set threshold value
            width = int(image.shape[1]*scale_percent)
            height = int(image.shape[0]*scale_percent)

            dim = (width,height)
            resized = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)

            kernel_sharpening = np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
            sharpened = cv2.filter2D(resized,-1,kernel_sharpening)

            gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
            inv = 255-gray
            gauss = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)

            def dodge(image,mask):
                return cv2.divide(image,255-mask,scale=256)

            output = dodge(gray,gauss)

        # send image to postman
        I = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) # convert image formate
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'jpeg')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/jpeg') 
    return output

if __name__ == '__main__':
    app.run()