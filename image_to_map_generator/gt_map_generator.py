"""Generate ground truth maps from an input image and a reference map.

Refer to README.md for detailed usage.

DISCLAIMER: This code is provided as-is and has only been tested for the example use cases provided in the README.md.
            It is not guaranteed to work for all use cases and may require modification to work for your use case.
            The code is provided for reference only and is not guaranteed to be the most efficient implementation.

WARNING: 1) This code is not guaranteed to be bug-free.  It is the user's responsibility to verify that the generated
         map is correct especially if you are testing with your own custom maps.
         2) This code is ONLY to be used for testing purposes during the development of your code. IT IS CONSIDERED
         CHEATING if you are caught using this code to generate your ground truth maps for during any milestone marking.

Adapted from code written by Kal Backman, 2022
Last modified by Calvin Vong, 2023
"""
import numpy as np
import cv2
import sys
import json
import os
from dataclasses import dataclass
import yaml

from PyQt5.QtCore import Qt
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from numpy import ndarray


@dataclass
class Parameters:
    DesiredGroundTruth: str # File path to the reference map file
    InputImage: str             # File path to the input image
    SaveImageName: str # File path to the saved map location

    ArenaSize: ndarray # [Width, Height] (m)

    ImageDisplayResolution: list # [Width, Height] pixels (not actual image resolution, just resolution to display for GUI)
    ArenaImageResolution: float

    # Object names and colours
    Objects: list[list]

    ArUcoBlockSize: float

    # Rendering stuff
    BorderSize: int
    GridSize: int
    ReferenceErrorLineSize: int
    ObjectPointSize: int
    ArUcoFontSize: float
    ArUcoFontThickness: int

    MaxClickRadius: int


class MainView(QMainWindow):
    def __init__(self):
        super(MainView, self).__init__()

        # init params
        self.loadParams()

        self.LoadData()

        self.BorderPoints = [np.array([0.0, 0.0], dtype=np.float32), np.array([self._params.ImageDisplayResolution[0] - 1, 0.0], dtype=np.float32),
                             np.array([self._params.ImageDisplayResolution[0] - 1, self._params.ImageDisplayResolution[1] - 1], dtype=np.float32), np.array([0.0, self._params.ImageDisplayResolution[1] - 1], dtype=np.float32)]
        self.SelectedBorderPoint = -1
        self.SelectedObject = -1
        self.MouseClicked = False

        self.ColourMap = cv2.applyColorMap(np.linspace(0, 255, 256, dtype=np.uint8), cv2.COLORMAP_RAINBOW).reshape([-1, 3])
        self.BorderColour = np.uint8(0)
        self.GridColour = np.uint8(255)
        self.ArUcoColour = np.uint8(128)
        self.GridDiv = 4

        self.setContentsMargins(0, 10, 0, 0)
        ColourPalette = self.palette()
        ColourPalette.setColor(self.backgroundRole(), QtGui.QColor(43, 43, 43))
        self.setPalette(ColourPalette)

        self.InputImageLbl = QLabel(self)
        self.InputImageLbl.mousePressEvent = self.InputImageMouseClick
        self.InputImageLbl.mouseReleaseEvent = self.InputImageRelease
        self.InputImageLbl.mouseMoveEvent = self.InputImageMove

        self.TransformedInputImageLbl = QLabel(self)
        self.TransformedInputImageLbl.mousePressEvent = self.TransformedImageMouseClick
        self.TransformedInputImageLbl.mouseReleaseEvent = self.TransformedImageRelease
        self.TransformedInputImageLbl.mouseMoveEvent = self.TransformedImageMove

        self.ReferenceArenaLbl = QLabel(self)
        self.EstimatedArenaLbl = QLabel(self)

        self.InputImageTitleLbl = QLabel("Input image")
        self.InputImageTitleLbl.setStyleSheet("color:rgb(220, 220, 220); font-size: 20px")
        self.TransformedInputImageTitleLbl = QLabel("Transformed input image")
        self.TransformedInputImageTitleLbl.setStyleSheet("color:rgb(220, 220, 220); font-size: 20px")
        self.ReferenceArenaTitleLbl = QLabel("Reference map")
        self.ReferenceArenaTitleLbl.setStyleSheet("color:rgb(220, 220, 220); font-size: 20px")
        self.EstimatedArenaTitleLbl = QLabel("Estimated map")
        self.EstimatedArenaTitleLbl.setStyleSheet("color:rgb(220, 220, 220); font-size: 20px")

        self.SaveBtn = QPushButton("Save", self)
        self.SaveBtn.setStyleSheet("background-color:rgb(220, 220, 220)")
        self.SaveBtn.setFocusPolicy(Qt.NoFocus)
        self.SaveBtn.clicked.connect(self.SaveBtnClicked)

        self.mainGrid = QGridLayout()

        self.mainGrid.addWidget(self.InputImageTitleLbl, 0, 0, 1, 20)
        self.mainGrid.addWidget(self.InputImageLbl, 1, 0, 20, 20)
        self.mainGrid.addWidget(self.TransformedInputImageTitleLbl, 21, 0, 1, 20)
        self.mainGrid.addWidget(self.TransformedInputImageLbl, 22, 0, 20, 20)
        self.mainGrid.addWidget(self.ReferenceArenaTitleLbl, 0, 20, 1, 20)
        self.mainGrid.addWidget(self.ReferenceArenaLbl, 1, 20, 20, 20)
        self.mainGrid.addWidget(self.EstimatedArenaTitleLbl, 21, 20, 1, 20)
        self.mainGrid.addWidget(self.EstimatedArenaLbl, 22, 20, 20, 20)
        self.mainGrid.addWidget(self.SaveBtn, 42, 0, 1, 40)


        self.mainGridWidget = QWidget()
        self.mainGridWidget.setLayout(self.mainGrid)
        self.setCentralWidget(self.mainGridWidget)

        self.RenderImages()

    def LoadData(self):

        try:
            self.LoadedImage = cv2.imread(self._params.InputImage)
        except Exception as e:
            print("Failed to load the input image!")
            print(e)
            self.LoadedImage = np.zeros([self._params.ImageDisplayResolution[1], self._params.ImageDisplayResolution[0], 3], dtype=np.uint8)

        try:
            with open(self._params.DesiredGroundTruth) as File:
                Lines = File.readlines()
                String = ""
                for Line in Lines:
                    String += Line
                self.ReferenceMap = json.loads(String)

        except Exception as e:
            print("Failed to load the map!")
            print(e)
            self.ReferenceMap = {}

        self.ReferenceObjects = []
        self.ReferenceArUcos = []

        for Key in self.ReferenceMap:

            SplitText = Key.split('aruco')
            if len(SplitText) > 1:

                ID = SplitText[1].split("_")
                if len(ID) > 1:
                    ID = int(ID[0])
                else:
                    ID = int(SplitText[1])

                Position = np.array([-self.ReferenceMap[Key]["x"], self.ReferenceMap[Key]["y"]])
                self.ReferenceArUcos.append([ID, Position])

            else:
                for ObjectIndex, Object in enumerate(self._params.Objects):
                    SplitText = Key.split(Object[0])

                    if len(SplitText) > 1:
                        ID = int(SplitText[1].split("_")[1])
                        Position = np.array([-self.ReferenceMap[Key]["x"], self.ReferenceMap[Key]["y"]])

                        self.ReferenceObjects.append([ID, Position, ObjectIndex])


        self.EstimatedArUcos = []
        self.EstimatedObjects = []
        for i in range(len(self.ReferenceArUcos)):
            self.EstimatedArUcos.append([self.ReferenceArUcos[i][0], self.ReferenceArUcos[i][1].copy()])

        for i in range(len(self.ReferenceObjects)):
            self.EstimatedObjects.append([self.ReferenceObjects[i][0], self.ReferenceObjects[i][1].copy(), self.ReferenceObjects[i][2]])

    def RenderImages(self):

        ###### Render input image ######
        InputImage = self.LoadedImage.copy()
        InputImage = cv2.resize(InputImage, (self._params.ImageDisplayResolution[0], self._params.ImageDisplayResolution[1]))

        cv2.line(InputImage, (int(self.BorderPoints[0][0]), int(self.BorderPoints[0][1])), (int(self.BorderPoints[1][0]), int(self.BorderPoints[1][1])), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(InputImage, (int(self.BorderPoints[1][0]), int(self.BorderPoints[1][1])), (int(self.BorderPoints[2][0]), int(self.BorderPoints[2][1])), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(InputImage, (int(self.BorderPoints[2][0]), int(self.BorderPoints[2][1])), (int(self.BorderPoints[3][0]), int(self.BorderPoints[3][1])), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(InputImage, (int(self.BorderPoints[3][0]), int(self.BorderPoints[3][1])), (int(self.BorderPoints[0][0]), int(self.BorderPoints[0][1])), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)

        InputImage = np.flip(InputImage, axis=2).copy()
        QtImage = QtGui.QImage(InputImage.data, InputImage.shape[1], InputImage.shape[0], InputImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.InputImageLbl.setPixmap(QtGui.QPixmap(QtImage))
        ###### Render input image ######


        ###### Render transformed image ######

        ReferencePoints = np.array([[0.0, 0.0],
                                    [self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1, 0],
                                    [self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1, self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1],
                                    [0.0, self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1]
                                    ], dtype=np.float32)

        ScaleX = self.LoadedImage.shape[1] / self._params.ImageDisplayResolution[0]
        ScaleY = self.LoadedImage.shape[0] / self._params.ImageDisplayResolution[1]
        TransformedBoarderPoints = np.array([[self.BorderPoints[0][0] * ScaleX, self.BorderPoints[0][1] * ScaleY],
                                             [self.BorderPoints[1][0] * ScaleX, self.BorderPoints[1][1] * ScaleY],
                                             [self.BorderPoints[2][0] * ScaleX, self.BorderPoints[2][1] * ScaleY],
                                             [self.BorderPoints[3][0] * ScaleX, self.BorderPoints[3][1] * ScaleY]], dtype=np.float32)

        Transform = cv2.getPerspectiveTransform(TransformedBoarderPoints, ReferencePoints)
        TransformedInputImage = cv2.warpPerspective(self.LoadedImage, Transform,
                                                   (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution)))
        CleanTransformedInputImage = TransformedInputImage.copy()

        XGridSize = self._params.ArenaSize[0] * self._params.ArenaImageResolution / self.GridDiv
        YGridSize = self._params.ArenaSize[1] * self._params.ArenaImageResolution / self.GridDiv
        for i in range(self.GridDiv - 1):
            cv2.line(TransformedInputImage, (int((i + 1) * XGridSize), int(0)), (int((i + 1) * XGridSize), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]), int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)
            cv2.line(TransformedInputImage, (int(0), int((i + 1) * YGridSize)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int((i + 1) * YGridSize)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]), int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)

        cv2.line(TransformedInputImage, (int(0), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(TransformedInputImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(TransformedInputImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(TransformedInputImage, (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)

        for Object in self.EstimatedObjects:
            ID = Object[0]
            Position = Object[1]
            Colour = self._params.Objects[Object[2]][1]

            cv2.circle(TransformedInputImage, (int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution), int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)), self._params.ObjectPointSize, Colour, -1)

        for ArUco in self.EstimatedArUcos:
            ID = ArUco[0]
            Position = ArUco[1]

            cv2.rectangle(TransformedInputImage, (int(((Position[0] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                          (int(((Position[0] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                          (int(self.ColourMap[self.ArUcoColour, 0]), int(self.ColourMap[self.ArUcoColour, 1]), int(self.ColourMap[self.ArUcoColour, 2])), -1)

            TextSize, _ = cv2.getTextSize(f"{ID}", cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize, self._params.ArUcoFontThickness)
            cv2.putText(TransformedInputImage, f"{ID}", (
            int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution - TextSize[0] / 2),
            int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution + TextSize[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize,
            (int(255 - self.ColourMap[self.ArUcoColour, 0]), int(255 - self.ColourMap[self.ArUcoColour, 1]),
            int(255 - self.ColourMap[self.ArUcoColour, 2])), self._params.ArUcoFontThickness, cv2.LINE_AA)

        TransformedInputImage = cv2.addWeighted(CleanTransformedInputImage, 0.6, TransformedInputImage, 0.4, 1)
        DisplayImage = np.flip(TransformedInputImage, axis=2).copy()
        QtImage = QtGui.QImage(DisplayImage.data, DisplayImage.shape[1], DisplayImage.shape[0], DisplayImage.shape[1] * 3,
                               QtGui.QImage.Format_RGB888)
        self.TransformedInputImageLbl.setPixmap(QtGui.QPixmap(QtImage))
        ###### Render transformed image ######


        ###### Render arena reference image ######
        ArenaReferenceImage = np.zeros([int(self._params.ArenaSize[1] * self._params.ArenaImageResolution), int(self._params.ArenaSize[0] * self._params.ArenaImageResolution), 3], dtype=np.uint8)

        XGridSize = self._params.ArenaSize[0] * self._params.ArenaImageResolution / self.GridDiv
        YGridSize = self._params.ArenaSize[1] * self._params.ArenaImageResolution / self.GridDiv
        for i in range(self.GridDiv - 1):
            cv2.line(ArenaReferenceImage, (int((i + 1) * XGridSize), int(0)), (int((i + 1) * XGridSize), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]), int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)
            cv2.line(ArenaReferenceImage, (int(0), int((i + 1) * YGridSize)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int((i + 1) * YGridSize)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]), int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)

        cv2.line(ArenaReferenceImage, (int(0), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaReferenceImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaReferenceImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaReferenceImage, (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)

        for Object in self.ReferenceObjects:
            ID = Object[0]
            Position = Object[1]
            Colour = self._params.Objects[Object[2]][1]

            cv2.circle(ArenaReferenceImage, (int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution), int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)), self._params.ObjectPointSize, Colour, -1)

        for ArUco in self.ReferenceArUcos:
            ID = ArUco[0]
            Position = ArUco[1]

            cv2.rectangle(ArenaReferenceImage, (int(((Position[0] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                                               (int(((Position[0] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                                               (int(self.ColourMap[self.ArUcoColour, 0]), int(self.ColourMap[self.ArUcoColour, 1]), int(self.ColourMap[self.ArUcoColour, 2])), -1)

            TextSize, _ = cv2.getTextSize(f"{ID}", cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize, self._params.ArUcoFontThickness)
            cv2.putText(ArenaReferenceImage, f"{ID}", (int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution - TextSize[0] / 2), int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution + TextSize[1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize,
                                                      (int(255 - self.ColourMap[self.ArUcoColour, 0]), int(255 - self.ColourMap[self.ArUcoColour, 1]), int(255 - self.ColourMap[self.ArUcoColour, 2])), self._params.ArUcoFontThickness, cv2.LINE_AA)

        ArenaReferenceImage = np.flip(ArenaReferenceImage, axis=2).copy()
        QtImage = QtGui.QImage(ArenaReferenceImage.data, ArenaReferenceImage.shape[1], ArenaReferenceImage.shape[0], ArenaReferenceImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.ReferenceArenaLbl.setPixmap(QtGui.QPixmap(QtImage))
        ###### Render arena reference image ######

        ###### Render arena estimated image ######
        ArenaEstimateImage = np.zeros([int(self._params.ArenaSize[1] * self._params.ArenaImageResolution),
                                       int(self._params.ArenaSize[0] * self._params.ArenaImageResolution), 3],
                                       dtype=np.uint8)

        XGridSize = self._params.ArenaSize[0] * self._params.ArenaImageResolution / self.GridDiv
        YGridSize = self._params.ArenaSize[1] * self._params.ArenaImageResolution / self.GridDiv
        for i in range(self.GridDiv - 1):
            cv2.line(ArenaEstimateImage, (int((i + 1) * XGridSize), int(0)),(int((i + 1) * XGridSize), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]),                     int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)
            cv2.line(ArenaEstimateImage, (int(0), int((i + 1) * YGridSize)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int((i + 1) * YGridSize)), (int(self.ColourMap[self.GridColour, 0]), int(self.ColourMap[self.GridColour, 1]), int(self.ColourMap[self.GridColour, 2])), self._params.GridSize)

        cv2.line(ArenaEstimateImage, (int(0), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaEstimateImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(0)), (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaEstimateImage, (int(self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)
        cv2.line(ArenaEstimateImage, (int(0), int(self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1)), (int(0), int(0)), (int(self.ColourMap[self.BorderColour, 0]), int(self.ColourMap[self.BorderColour, 1]), int(self.ColourMap[self.BorderColour, 2])), self._params.BorderSize)

        for Index, Object in enumerate(self.EstimatedObjects):
            ID = Object[0]
            Position = Object[1]
            Colour = self._params.Objects[Object[2]][1]

            ReferencePosition = self.ReferenceObjects[Index][1]
            cv2.circle(ArenaEstimateImage, (int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution), int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)), self._params.ObjectPointSize, Colour, -1)
            cv2.line(ArenaEstimateImage, (int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution), int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)),
                                         (int((ReferencePosition[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution), int((ReferencePosition[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)),
            (255, 20, 20), self._params.ReferenceErrorLineSize)

        for Index, ArUco in enumerate(self.EstimatedArUcos):
            ID = ArUco[0]
            Position = ArUco[1]
            ReferencePosition = self.ReferenceArUcos[Index][1]

            cv2.rectangle(ArenaEstimateImage, (int(((Position[0] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) - self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                          (int(((Position[0] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution), (int(((Position[1] + self._params.ArenaSize[0] / 2) + self._params.ArUcoBlockSize / 2) * self._params.ArenaImageResolution))),
                          (int(self.ColourMap[self.ArUcoColour, 0]), int(self.ColourMap[self.ArUcoColour, 1]), int(self.ColourMap[self.ArUcoColour, 2])), -1)

            cv2.line(ArenaEstimateImage, (
            int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution),
            int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)),
                     (int((ReferencePosition[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution),
                      int((ReferencePosition[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution)),
                     (255, 20, 20), self._params.ReferenceErrorLineSize)

            TextSize, _ = cv2.getTextSize(f"{ID}", cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize, self._params.ArUcoFontThickness)
            cv2.putText(ArenaEstimateImage, f"{ID}", (
            int((Position[0] + self._params.ArenaSize[0] / 2) * self._params.ArenaImageResolution - TextSize[0] / 2),
            int((Position[1] + self._params.ArenaSize[1] / 2) * self._params.ArenaImageResolution + TextSize[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, self._params.ArUcoFontSize,
            (int(255 - self.ColourMap[self.ArUcoColour, 0]), int(255 - self.ColourMap[self.ArUcoColour, 1]),
             int(255 - self.ColourMap[self.ArUcoColour, 2])), self._params.ArUcoFontThickness, cv2.LINE_AA)

        ArenaEstimateImage = np.flip(ArenaEstimateImage, axis=2).copy()

        QtImage = QtGui.QImage(ArenaEstimateImage.data, ArenaEstimateImage.shape[1], ArenaEstimateImage.shape[0],
                               ArenaEstimateImage.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.EstimatedArenaLbl.setPixmap(QtGui.QPixmap(QtImage))
        ###### Render arena estimated image ######

    def InputImageMouseClick(self, Event):
        if Event.button() == QtCore.Qt.LeftButton:
            self.MouseClicked = True
            MousePosition = np.array([Event.pos().x(), Event.pos().y()])

            Distance = []
            for Point in self.BorderPoints:
                Distance.append(np.linalg.norm(MousePosition - Point))

            Index = np.argmin(Distance)
            if Distance[Index] < self._params.MaxClickRadius:
                self.SelectedBorderPoint = np.argmin(Distance)
            else:
                self.SelectedBorderPoint = -1

            self.SelectedObject = -1

    def InputImageRelease(self, Event):
        if Event.button() == QtCore.Qt.LeftButton:
            self.MouseClicked = False

    def InputImageMove(self, Event):

        if self.MouseClicked:

            MousePosition = np.array([Event.pos().x(), Event.pos().y()])
            if MousePosition[0] >= 0 and MousePosition[0] <= self._params.ImageDisplayResolution[0] - 1:
                if MousePosition[1] >= 0 and MousePosition[1] <= self._params.ImageDisplayResolution[1] - 1:
                    if self.SelectedBorderPoint >= 0:
                        self.BorderPoints[self.SelectedBorderPoint] = MousePosition
                        self.RenderImages()

    def TransformedImageMouseClick(self, Event):
        if Event.button() == QtCore.Qt.LeftButton:
            self.MouseClicked = True
            MousePosition = np.array([Event.pos().x(), Event.pos().y()])

            Distances = []
            for ArUco in self.EstimatedArUcos:
                Distances.append(np.linalg.norm(MousePosition - (ArUco[1] + self._params.ArenaSize / 2) * self._params.ArenaImageResolution))

            for Object in self.EstimatedObjects:
                Distances.append(np.linalg.norm(MousePosition - (Object[1] + self._params.ArenaSize / 2) * self._params.ArenaImageResolution))

            Index = np.argmin(Distances)
            if Distances[Index] < self._params.MaxClickRadius:
                self.SelectedObject = Index
            else:
                self.SelectedObject = -1

            self.SelectedBorderPoint = -1

    def TransformedImageRelease(self, Event):
        if Event.button() == QtCore.Qt.LeftButton:
            self.MouseClicked = False

    def TransformedImageMove(self, Event):

        if self.MouseClicked:

            MousePosition = np.array([Event.pos().x(), Event.pos().y()])
            if MousePosition[0] >= 0 and MousePosition[0] <= self._params.ArenaSize[0] * self._params.ArenaImageResolution - 1:
                if MousePosition[1] >= 0 and MousePosition[1] <= self._params.ArenaSize[1] * self._params.ArenaImageResolution - 1:
                    if self.SelectedObject >= 0:

                        if self.SelectedObject < len(self.EstimatedArUcos):
                            self.EstimatedArUcos[self.SelectedObject][1][0] = (MousePosition[0] - self._params.ArenaSize[0] * self._params.ArenaImageResolution / 2) / self._params.ArenaImageResolution
                            self.EstimatedArUcos[self.SelectedObject][1][1] = (MousePosition[1] - self._params.ArenaSize[1] * self._params.ArenaImageResolution / 2) / self._params.ArenaImageResolution

                        else:
                            self.EstimatedObjects[self.SelectedObject - len(self.EstimatedArUcos)][1][0] = (MousePosition[0] - self._params.ArenaSize[0] * self._params.ArenaImageResolution / 2) / self._params.ArenaImageResolution
                            self.EstimatedObjects[self.SelectedObject  - len(self.EstimatedArUcos)][1][1] = (MousePosition[1] - self._params.ArenaSize[1] * self._params.ArenaImageResolution / 2) / self._params.ArenaImageResolution

                        self.RenderImages()

    def SaveBtnClicked(self):
        EstimateMap = {}
        for ArUco in self.EstimatedArUcos:
            EstimateMap[f"aruco{ArUco[0]}_0"] = {"y": ArUco[1][1], "x": -ArUco[1][0]}

        for Object in self.EstimatedObjects:
            EstimateMap[f"{self._params.Objects[Object[2]][0]}_{Object[0]}"] = {"y": Object[1][1], "x": -Object[1][0]}

        with open(self._params.SaveImageName, "w") as file:
            json.dump(EstimateMap, file)
            print("Saved map!")
    
    def loadParams(self):
        cwd = os.path.dirname(os.path.abspath(sys.argv[0]))

        with open(cwd + '/config.yaml', 'r') as f:
            params = yaml.safe_load(f)

        arena_size = np.array([params['arena_size']['x'], params['arena_size']['y']])
        image_display_resolution = [params['image_display_res']['x'], params['image_display_res']['y']]
        arena_display_res = image_display_resolution[1] / arena_size[1]

        objects = []
        for fruit in params['fruits']:
            objects.append([fruit, params['fruits'][fruit]])

        self._params = Parameters(
            DesiredGroundTruth=cwd + '/reference_maps/' + params['reference_map'],
            InputImage=cwd + '/input_images/' + params['input_image'],
            SaveImageName=cwd + '/generated_ground_truth_maps/' + params['output_map_name'],
            ArenaSize=arena_size,
            ImageDisplayResolution=image_display_resolution,
            ArenaImageResolution=arena_display_res,
            Objects=objects,
            ArUcoBlockSize=params['ArUco_block_size'],
            BorderSize=params['border_size'],
            GridSize=params['grid_size'],
            ReferenceErrorLineSize=params['reference_error_line_size'],
            ObjectPointSize=params['object_point_size'],
            ArUcoFontSize=params['ArUco_font_size'],
            ArUcoFontThickness=params['ArUco_font_thickness'],
            MaxClickRadius=params['max_click_radius']
        )


if __name__ == '__main__':
    # Start interface
    app = QApplication(sys.argv)
    GUI = MainView()
    GUI.show()
    GUI.setFixedSize(GUI.size())
    sys.exit(app.exec_())