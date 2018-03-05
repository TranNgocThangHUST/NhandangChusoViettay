import numpy as np
from scipy.misc.pilutil import Image
import cv2
import scipy.misc as mi
from keras.models import model_from_json 

def reStoreModel():
    
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def testDigitFrom(direction, cnnModel):
    im = Image.open(direction).convert('L')
    im1 = im.crop((599, 601, 615, 623))
    # im1 = im.crop((590, 53, 610, 76))
    test = np.asarray(im1)
    test.setflags(write=1)
    test = cv2.resize(test, (28, 28))
    # img = np.zeros((20,20,3), np.uint8)
    

    # format picture
    for x in test:
        i = 0
        for y in x:
            t = 255 - y
            x[i] = t
            i += 1
    for x in test:
        i = 0
        for y in x:
            if y < 30:
                x[i] = 0
            i += 1
                
    # xóa đường kẻ ngang
    arr = []
    for x in test:
        i = 0
        a = 0
        if x[0] > 40:
            for y in x:
                if a < 70:
                    x[i] = 0
                    arr[i] = 0
                else:
                    arr[i] = x[i]
                print(arr)
                i += 1
                if i > 27:
                    break
                else:
                    a = arr[i]
                    print(a)
        else:
            arr.clear()
            for y in x:
                arr.append(y)
                i += 1
            print(arr)
            a = arr[0]
        print(a)
    print(i) 
# =============================================================================
#     # khôi phục trường hợp xóa phải con số
#     if i == 28:
#         j = 0
#         for x in test:
#            if j < 27:
#                j += 1
#                continue
#            else:
#                k = 0
#                l = 0
#                a = 0
#                b = 0
#                for y in x:
#                    if y > 0:
#                        a = k
#                        break
#                    k += 1
#                for y in x:
#                    if y > 0:
#                        b = l
#                    l += 1
#     print(a)
#     print(b)
#     cv2.line(test, (a, 27), (b, 27), (150, 0, 0), 1)
# =============================================================================
    cv2.line(test, (13, 27), (20, 27), (150, 0, 0), 2)
    print(test)
    
    # from PIL import Image
    # mi.imsave('test.jpg', test)
    # im is converted to a numpy ndarray
    # test = mi.fromimage(im)
    #test[test>170]=0 # để cho ảnh chuyển sang nền màu đen giống mẫu luyện
    
    test = test / 255.0
    test = test.reshape(-1,28,28,1)
    # predict results
    results = model.predict(test)
    # select the indix with the maximum probability
    results = np.argmax(results,axis = 1)
    # results = pd.Series(results,name="Label")
    return results
    
model = reStoreModel()
digit = testDigitFrom('input/pic.png', model)
print(digit)

# =============================================================================
# im = Image.open('input/pic1.png').convert('L')
# #2 im1 = im.crop((590, 53, 610, 76))
# #0 im1 = im.crop((613, 51, 632, 74))
# #4 im1 = im.crop((591, 78, 609, 101))
# #5 im1 = im.crop((615, 78, 634, 100))
# #2 im1 = im.crop((597, 135, 617, 153))

# =============================================================================

# =============================================================================
# im = Image.open('input/pic.png').convert('L')
# #2 im1 = im.crop((590, 53, 610, 76))
# #0 im1 = im.crop((613, 51, 632, 74))
# #4 im1 = im.crop((591, 78, 609, 101))
# #5 im1 = im.crop((615, 78, 634, 100))
# #2 im1 = im.crop((597, 135, 617, 153))
# #5 im1 = im.crop((622, 133, 642, 153))
# #5 im1 = im.crop((598, 162, 619, 180))
# #5 im1 = im.crop((621, 158, 641, 179))
# #6 im1 = im.crop((595, 185, 613, 209))
# #0 im1 = im.crop((622, 186, 639, 205))

# #CS1_2.7 im1 = im.crop((598, 212, 620, 234))

# #0 im1 = im.crop((625, 211, 644, 231))
# #3 im1 = im.crop((600, 235, 619, 258))
# #5 im1 = im.crop((626, 235, 644, 257))
# #3 im1 = im.crop((603, 262, 622, 283))
# #0 im1 = im.crop((631, 262, 649, 283))
# #6 im1 = im.crop((602, 288, 622, 311))
# #5 im1 = im.crop((626, 287, 642, 310))
# #2 im1 = im.crop((599, 340, 621, 364))
# #0 im1 = im.crop((627, 340, 648, 362))
# #3 im1 = im.crop((594, 367, 613, 388))
# #5 im1 = im.crop((622, 365, 639, 388))
# #4 im1 = im.crop((608, 392, 628, 415))
# #0 im1 = im.crop((630, 393, 646, 414))
# #CS2_2.7 im1 = im.crop((590, 417, 610, 440))
# #0 im1 = im.crop((614, 417, 634, 439))
# #5 im1 = im.crop((607, 444, 629, 466))
# #5 im1 = im.crop((593, 472, 613, 493))
# #0 im1 = im.crop((614, 469, 635, 493))
# #3 im1 = im.crop((605, 497, 621, 518))
# #0 im1 = im.crop((625, 497, 642, 518))
# #5 im1 = im.crop((588, 521, 608, 544))
# #5 im1 = im.crop((615, 521, 634, 543))
# #7 im1 = im.crop((596, 573, 619, 596))
# #0 im1 = im.crop((622, 574, 641, 595))
# #CS3_8.6 im1 = im.crop((599, 601, 615, 623))
# #5 im1 = im.crop((622, 600, 638, 622))
# #5 im1 = im.crop((598, 628, 616, 650))
# #5 im1 = im.crop((619, 626, 639, 648))
# #9 im1 = im.crop((607, 654, 627, 676))
# #4 im1 = im.crop((597, 681, 618, 702))
# #0 im1 = im.crop((622, 680, 641, 701))
# #5 im1 = im.crop((607, 704, 626, 727))
# #0 im1 = im.crop((630, 704, 648, 728))
# #1 im1 = im.crop((592, 732, 613, 753))
# #0 im1 = im.crop((621, 733, 639, 754))
# #6 im1 = im.crop((602, 757, 623, 779))
# #0 im1 = im.crop((627, 758, 644, 780))
# #5 im1 = im.crop((600, 784, 623, 806))
# #5 im1 = im.crop((627, 784, 649, 806))
# =============================================================================