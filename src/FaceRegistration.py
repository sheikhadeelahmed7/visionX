from FaceRegistrationBase import FaceRegistrarBase

import numpy as np
import face_recognition
import filetype
import os
import cv2

class FaceRegistrar(FaceRegistrarBase):

    def __init__(self):

        self.status = None
        self.statusDescription = []

        self._image_bgr = None
        self._image_rgb = None
        self._outputDetections = None
        self._crop = None
        self._faceDetector = None
        self._facialPoints = []
        self._eyePoints = None
        self._lmDetectorSlopeThresh = 0.16
        self._iouThresholdMin = 2.0
        self._iouThresholdMax = 18.0
        self._edgeThreshold = 0.03
        self._imageHeightCheck = 1080
        self._imageWidthCheck = 1920
        self._accepted_extensions = ["jpg", "jpeg"]
        self._illuminationThreshold = 105
        self._cropIlluminationThreshold = 65
        self._illuminationDiffThreshold = 55
        self._blurThreshold = 75
        self._slopeThresh = 0.17


    def BGR2RGB(self):
        self._image_rgb = cv2.cvtColor(self._image_bgr,cv2.COLOR_RGB2BGR)

    def getCrop(self):
        crop = self._image_bgr[int(self._outputDetections[0][0]): int(self._outputDetections[0][2]),
               int(self._outputDetections[0][3]): int(self._outputDetections[0][1])]
        return crop

    def calcIou(self,bboxes1, bboxes2):
        if not (bboxes1.shape[0] > 0 and bboxes2.shape[0] > 0):
            return
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
        boxAArea = (x12 - x11) * (y12 - y11)
        boxBArea = (x22 - x21) * (y22 - y21)
        iou = interArea / (1 * (boxAArea + np.transpose(boxBArea) - interArea + 0.000001))
        return iou * 100

    def getFacialPoints(self):
        if (face_recognition.face_landmarks(self._image_rgb)) != []:
            self._facialPoints = (face_recognition.face_landmarks(self._image_rgb))[0]
            return True
        else:
            return False

    def centreEyePoints(self):
        right_eye_x = int((self._facialPoints['right_eye'][0][0] + self._facialPoints['right_eye'][3][0]) / 2)
        right_eye_y = int((self._facialPoints['right_eye'][0][1] + self._facialPoints['right_eye'][3][1]) / 2)
        left_eye_x = int((self._facialPoints['left_eye'][0][0] + self._facialPoints['left_eye'][3][0]) / 2)
        left_eye_y = int((self._facialPoints['left_eye'][0][1] + self._facialPoints['left_eye'][3][1]) / 2)

        self._eyePoints = [(right_eye_x, right_eye_y), (left_eye_x, left_eye_y)]

    def checkLandmarkSlope(self):

        slope = (self._eyePoints[0][1] - self._eyePoints[1][1]) / (self._eyePoints[0][0] - self._eyePoints[1][0])
        if (slope > (-1 * self._slopeThresh)) and (slope < self._slopeThresh):  # FIXME find threshold empirically
            return True
        return False

    def fileTypeCheck(self, IMG_PATH):

        #if image path exists
        if os.path.exists(IMG_PATH):
            #getting mime information of file
            file_type, ext = filetype.guess_mime(IMG_PATH).split('/')

            if file_type == 'image':
                #loading Image in RGB and BGR
                self._image_bgr = cv2.imread(IMG_PATH)
                self.BGR2RGB()

                if ext.lower() in self._accepted_extensions:
                    if self._image_bgr is not None:
                        if (self._image_bgr.shape[1] == self._imageWidthCheck) and (
                                self._image_bgr.shape[0] == self._imageHeightCheck):

                            if 'FILE TYPE CHECK PASSED' in self.statusDescription:
                                return True
                            else:
                                self.statusDescription.append('FILE TYPE CHECK PASSED')
                                return True
                        else:
                            if ((self._image_bgr.shape[1] / self._imageWidthCheck) > 0.5) \
                                    and (
                                    (self._image_bgr.shape[0] / self._imageHeightCheck) > 0.6):
                                if 'FILE TYPE CHECK PASSED' in self.statusDescription:
                                    pass
                                else:
                                    self.statusDescription.append('FILE TYPE CHECK PASSED')
                                return True,

                            else:
                                self.statusDescription.append('DIMENSION ISSUE')
                                return False
                    else:
                        self.statusDescription.append('IMAGE NOT LOADED')
                        return False
                else:
                    self.statusDescription.append('FILE EXTENSION ERROR')
                    return False
            else:
                self.statusDescription.append('WRONG FILE: ' + file_type)
                return False
        else:
            self.statusDescription.append('NO FILE EXISTS')
            return False

    def faceDetectionCheck(self):

        self._outputDetections = face_recognition.face_locations(self._image_rgb)

        if len(self._outputDetections) == 1:

            self._crop = self.getCrop()

            ymin = int(self._outputDetections[0][0])
            ymax = int(self._outputDetections[0][2])
            xmin = int(self._outputDetections[0][3])
            xmax = int(self._outputDetections[0][1])

            outputDetections = [(xmin, ymin, xmax, ymax)]

            iouDetectedCrop = self.calcIou(np.array(outputDetections),
                                      np.array([[0, 0, self._image_bgr.shape[1], self._image_bgr.shape[0]]]))

            if iouDetectedCrop >= self._iouThresholdMin and iouDetectedCrop <= self._iouThresholdMax:
                bboxNormalized = outputDetections / np.array([self._image_bgr.shape[1], self._image_bgr.shape[0],
                                                              self._image_bgr.shape[1], self._image_bgr.shape[0]])
                if (bboxNormalized[0][0] > self._edgeThreshold) and \
                        (bboxNormalized[0][0] < (1 - self._edgeThreshold)) and \
                        (bboxNormalized[0][1] > self._edgeThreshold) and \
                        (bboxNormalized[0][1] < (1 - self._edgeThreshold)) and \
                        (bboxNormalized[0][0] > self._edgeThreshold) and \
                        (bboxNormalized[0][0] < (1 - self._edgeThreshold)) and \
                        (bboxNormalized[0][1] > self._edgeThreshold) and \
                        (bboxNormalized[0][1] < (1 - self._edgeThreshold)):

                    self.statusDescription.append('FACE DETECTION CHECK PASSED')
                    return True
                else:
                    self.statusDescription.append('FACES ON EDGE')
                    return False
            else:
                self.statusDescription.append('FACE SIZE ISSUE')
                return False
        else:
            if self._outputDetections == []:
                self.statusDescription.append('NO FACES FOUND')
                return False
            else:
                self.statusDescription.append('MULTI FACES DETECTED')
                return False

    def landMarkCheck(self):

        if self.getFacialPoints():

            if self._facialPoints != []:

                self.centreEyePoints()

                if self.checkLandmarkSlope():
                    self.statusDescription.append('FACIAL LANDMARK CHECK PASSED')
                    return True
                else:
                    self.statusDescription.append('FACE ORIENTATION ISSUE')
                    return False
            else:
                self.statusDescription.append('LANDMARK NOT FOUNDED')
                return False
        else:
            self.statusDescription.append('LANDMARK NOT FOUNDED')
            return False

    def blurCheck(self):

        laplacian_var = cv2.Laplacian(self._crop, cv2.CV_64F).var()
        if laplacian_var > self._blurThreshold:
            self.statusDescription.append('BLUR CHECK PASSED')
            return True
        else:
            self.statusDescription.append('BLUR ERROR')
            return False

    def illuminationCheck(self):

        hsv = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        illuminationMean = v.mean()

        if illuminationMean > self._illuminationThreshold:  # light condition in image is okay

            if self._crop is not None:

                hsvDetectedCrop = cv2.cvtColor(self._crop, cv2.COLOR_BGR2HSV)
                detectedCropV = hsvDetectedCrop[:, :, 2]
                detectedCropIlluminationMean = detectedCropV.mean()

                if detectedCropIlluminationMean > self._cropIlluminationThreshold:
                    h, w, _ = self._crop.shape
                    istHalf = self._crop[:, :int(w / 2), :]
                    secHalf = self._crop[:, int(w / 2):, :]
                    istHalfHSV = cv2.cvtColor(istHalf, cv2.COLOR_BGR2HSV)
                    secHalfHSV = cv2.cvtColor(secHalf, cv2.COLOR_BGR2HSV)
                    istHalfVMean = istHalfHSV[:, :, 2].mean()
                    secHalfMean = secHalfHSV[:, :, 2].mean()
                    if abs(secHalfMean - istHalfVMean) < self._illuminationDiffThreshold:

                        self.statusDescription.append('ILLUMINATION CHECK PASSED')
                        return True
                    else:
                        self.statusDescription.append('FACE POSITION ERROR')
                        return False
                else:
                    self.statusDescription.append('FACE ILLUMINATION ERROR')
                    return False
            else:
                pass
        else:
            self.statusDescription.append('IMAGE ILLUMINATION ERROR')
            return False

    def getstatus(self,IMG_PATH):

        if self.fileTypeCheck(IMG_PATH):
            if self.faceDetectionCheck():
                if self.landMarkCheck():
                    if self.blurCheck():
                        if self.illuminationCheck():
                            self.status = "ACCEPTED"
                            return self.status , self.statusDescription
                        else:
                            self.status = "REJECTED"
                            return self.status , self.statusDescription
                    else:
                        self.status = "REJECTED"
                        return self.status, self.statusDescription
                else:
                    self.status = "REJECTED"
                    return self.status, self.statusDescription
            else:
                self.status = "REJECTED"
                return self.status, self.statusDescription
        else:
            self.status = "REJECTED"
            return self.status, self.statusDescription


