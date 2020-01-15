#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import cv2
import os

def findMaxIndex_GT(c_dict):
    maxIndex = -1;
    for key, val in c_dict.items():
        if(val[0] > maxIndex):
            maxIndex = val[0]
    return maxIndex


def findMaxIndex_pred(c_dict):
    maxIndex = -1;
    for key, val in c_dict.items():
        if(val > maxIndex):
            maxIndex = val
    return maxIndex


def findCorrect(c_matrix, T):
    correct = 0
    for i in range(c_matrix.shape[0]):
        
        found = False;
        
        for j in range(c_matrix.shape[1]):
            
            if ((c_matrix[i][j]/c_matrix[i].sum()) > 1-T):
                
                        flag = False;
                        for k in range(c_matrix.shape[0]):
                            if (k == i):
                                continue
                            if ((c_matrix[k][j]/c_matrix[:,j].sum()) >= T):
                                flag = True;
                                
                        if (flag == False):
                            found = True;
                            break;
        if(found == True):
            correct +=1;
    return correct;
            

def findPartialDetection(c_matrix, T):
    partial = 0
    for i in range(c_matrix.shape[0]):
        count = 0
        for j in range(c_matrix.shape[1]):
            
            if (((c_matrix[i][j]/c_matrix[i].sum()) < 1-T) and ((c_matrix[i][j]/c_matrix[i].sum()) > T)):
                count += 1
        if (count == 1):
            partial += 1;
    return partial

def findMissed(c_matrix, T):
    missed = 0
    for i in range(c_matrix.shape[0]):

        count = 0
        for j in range(c_matrix.shape[1]):
            
            if (((c_matrix[i][j]/c_matrix[i].sum()) < T)):
                count += 1
        if (count == c_matrix.shape[1]):
            missed += 1;
    return missed;

def findOverSegmented(c_matrix, T):
    over = 0
    for i in range(c_matrix.shape[0]):
        count = 0;
        for j in range(c_matrix.shape[1]):
            if (((c_matrix[i][j]/c_matrix[i].sum()) < 1-T) and ((c_matrix[i][j]/c_matrix[i].sum()) > T)):
                count += 1
        if (count > 1):
            over += 1;
    return over


def findUnderSegmented(c_matrix, T):
    under = 0
    for j in range(c_matrix.shape[1]):
        count = 0;
        for i in range(c_matrix.shape[0]):
            if (((c_matrix[i][j]/c_matrix[:,j].sum()) < 1-T) and ((c_matrix[i][j]/c_matrix[:,j].sum()) > T)):
                count += 1
        if (count > 1):
            under += 1;
    return under
            

def findFalsePositive(c_matrix, T):
    fp = 0
    for j in range(c_matrix.shape[1]):
        count = 0;
        for i in range(c_matrix.shape[0]):
            if (((c_matrix[i][j]/c_matrix[:,j].sum()) < T)):
                count += 1
        if (count == c_matrix.shape[0]):
            fp += 1;
    return fp


with open('./color_dict/0140_007.pickle', 'rb') as handle:
    color_dict_gt = pickle.load(handle)
print("loading")

gt_img = cv2.imread("./image_gt/0140_007.png", -1)
prediction = cv2.imread("./output/final_documents/0140_007.png", -1)
color_dict_prediction = np.load("./output/color_dicts/0140_007.png.npy").item()

print("GT SHAPE", gt_img.shape)
print("Prediction shape", prediction.shape)



c_matrix = np.zeros((findMaxIndex_GT(color_dict_gt)+1, findMaxIndex_pred(color_dict_prediction)+1))

for i in range(gt_img.shape[0]):
    
    for j in range(gt_img.shape[1]):
        
        if ((gt_img[i,j][0] == 0) and (gt_img[i,j][1] == 0) and (gt_img[i,j][2] == 0)):
            continue
        if ((prediction[i,j][0] == 0) and (prediction[i,j][1] == 0) and (prediction[i,j][2] == 0)):
            continue
        
        gt_index = color_dict_gt[tuple(gt_img[i,j])][0]
        prediction_index = color_dict_prediction[tuple(prediction[i,j])]
        
        c_matrix[gt_index, prediction_index] += 1
print(c_matrix)
        
        
# for i in range(gt_img.shape[0]):
#     for j in range(gt_img.shape[1]):
#         if ((gt_img[i,j][0] == 200) and (gt_img[i,j][1] == 0) and (gt_img[i,j][2] == 0)):
#             print("FOUND")
            

total_G = 0
total_S = 0
total_correct = 0;
total_partial = 0;
total_over = 0;
total_under = 0;
total_fp = 0
total_missed = 0;
img_no = 1;
for file in os.listdir("./color_dict_row/"):
    
    print("IMAGE:", img_no)
    print(file)
    with open('./color_dict_row/'+file,'rb') as handle:
        color_dict_gt = pickle.load(handle)
    
    img_name = file.strip().replace(".pickle", ".png")
    prediction_color_dict_name = file.strip().replace(".pickle", ".png.npy")

    gt_img = cv2.imread("./image_gt_row/"+img_name, -1)
    prediction = cv2.imread("./row_results/final_documents/"+img_name, -1)
    color_dict_prediction = np.load("./row_results/color_dicts/"+prediction_color_dict_name).item()


    c_matrix = np.zeros((findMaxIndex_GT(color_dict_gt)+1, findMaxIndex_pred(color_dict_prediction)+1))

    for i in range(gt_img.shape[0]):

        for j in range(gt_img.shape[1]):

            if ((gt_img[i,j][0] == 0) and (gt_img[i,j][1] == 0) and (gt_img[i,j][2] == 0)):
                continue
            if ((prediction[i,j][0] == 0) and (prediction[i,j][1] == 0) and (prediction[i,j][2] == 0)):
                continue
                
            gt_index = color_dict_gt[tuple(gt_img[i,j])][0]
            prediction_index = color_dict_prediction[tuple(prediction[i,j])]
            

            c_matrix[gt_index, prediction_index] += 1
    
    # c_matrix has been created
    print(c_matrix)
    print(c_matrix.shape)
    np.save("./row_matrices/"+ file.replace(".pickle", ".npy"), c_matrix)
    
    total_G += c_matrix.shape[0];
    total_S += c_matrix.shape[1];
    
    print("CURRENT_CORRECT:", findCorrect(c_matrix,0.1))
    total_correct += findCorrect(c_matrix, 0.1)
    total_partial += findPartialDetection(c_matrix, 0.1)
    total_under += findUnderSegmented(c_matrix, 0.1)
    total_over += findOverSegmented(c_matrix, 0.1)
    total_missed += findMissed(c_matrix, 0.1)
    total_fp += findFalsePositive(c_matrix, 0.1)
    
    print("Correct:",  (total_correct/total_G) * 100)

    
    img_no += 1;

correct_pc = (total_correct/total_G) * 100;
partial_pc = (total_partial/total_G) * 100;
under_pc = (total_under/total_S) * 100;
over_pc = (total_over/total_G) * 100;
missed_pc = (total_missed/total_G) * 100;
fp_pc = (total_fp/total_S) * 100;

print("=== RESULTS ===")
print("Correct:", correct_pc)
print("Partial:", partial_pc)
print("Undersegmented:", under_pc)
print("Oversegmented:", over_pc)
print("Missed:",missed_pc)
print("False Positive:", fp_pc)

print(findCorrect(c_matrix, 0.1))
print(findPartialDetection(c_matrix, 0.1))
print(findUnderSegmented(c_matrix, 0.1))
print(findOverSegmented(c_matrix, 0.1))
print(findMissed(c_matrix, 0.1))
print(findFalsePositive(c_matrix, 0.1))

(c_matrix[1][1]/c_matrix[1].sum())

c_dict = np.load("./output/fixed_lines/color_dicts/9563_061.png.npy").item()

c_dict

