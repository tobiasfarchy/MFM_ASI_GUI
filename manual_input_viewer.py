# -*- coding: utf-8 -*-
"""
@author: tfarchy

Packages to install: puzzlepiece, numpy, pyqtgraph, opencv-python, scipy, PyQt5
"""

import file_viewer as fv
import puzzlepiece as pzp
from importlib import reload
import numpy as np
from importlib import reload
import os
import sys
import pyqtgraph as pg
from functools import partial as fpartial
from cv2 import getRotationMatrix2D, warpAffine
from scipy.signal import find_peaks
from PyQt5.QtWidgets import QApplication, QFileDialog, QComboBox, QGraphicsProxyWidget
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import QRectF

def save_config_helper(bars, filename):
    np.save(filename + '_xbars.npy', np.array(bars[0]), allow_pickle = False)
    np.save(filename + '_ybars.npy', np.array(bars[1]), allow_pickle = False)

def iround(num):
    return int(np.round(num))

def clean_peaks(data_raw, quant = 0.8, dist: int = 8):
    ns = np.arange(len(data_raw))
    fit = np.polyfit(ns, data_raw, deg = 1)
    data = data_raw - fit[0]*ns - fit[1]
    peak_i, peak_h = find_peaks(data, height = (np.quantile(data, quant), np.max(data) + 1.), distance = dist)
    return peak_i

class ModeView(fv.FileView):
    def define_params(self):
        rot = pzp.param.spinbox(self, "Rotation", 0)(None)
        x_stretch = pzp.param.spinbox(self, "x_stretch", 0)(None)
        y_stretch = pzp.param.spinbox(self, "y_stretch", 0)(None)
        rot.changed.connect(self.rot_image)
        pzp.param.readout(self, 'Latest error')(None)
        self.rect = None

    def define_actions(self):
        @pzp.action.define(self, "Generate ASI Input")
        def generate_checkerboard(self):
            # Read horizontal and vertical lines
            y, x = int(np.round(self.crosshair_h.value())), int(np.round(self.crosshair_v.value()))
    
            # Predict SI dimensions
            row_raw, col_raw = self.rot_AFM[y], [self.rot_AFM[i][x] for i in range(len(self.rot_AFM))]
            peaks_x, peaks_y = clean_peaks(data_raw = row_raw, quant = 0.8), clean_peaks(data_raw = col_raw, quant = 0.8)
            image_size = (peaks_x[-1] - peaks_x[0], peaks_y[-1] - peaks_y[0])
            self.bar_width = np.mean([peaks_x[1] - peaks_x[0], peaks_y[1] - peaks_y[0]])
            num_x, num_y = len(peaks_x), len(peaks_y)
            self.origin = min(peaks_x), min(peaks_y)

            print(num_x, num_y)
            
            self.populate_checkerboard(rows = num_y, cols = num_x, image_size = image_size)

        @pzp.action.define(self, "Default mag")
        def default_checkerboard(self):
            if len(self.combos) > 0:
                for combo_lst in self.combos:
                    for combo in combo_lst:
                        combo.setCurrentIndex(0)
            else:
                self.params['Latest error'].set_value('Generate checkerboard first')
        
        @pzp.action.define(self, "Save Configuration ")
        def save_config(self):
            if len(self.combos) > 0:
                xbars = []
                ybars = []
                i = 0
                for combo_lst in self.combos:
                    bars_lst = []
                    for combo in combo_lst:
                        bars_lst.append(combo.currentIndex())
                    if i%2 == 0:
                        xbars.append(bars_lst)
                    else:
                        ybars.append(bars_lst)
                    i += 1
            else:
                self.params['Latest error'].set_value('Generate checkerboard first')

            # if 'saved_scans' not in os.listdir(os.getcwd()):
            #     os.mkdir(os.getcwd() + '//saved_scans')
            fname = (self.filename[:-len("_MFM.txt")] +
                  f"_rot{self.params['Rotation'].value}")
            save_config_helper((xbars, ybars), fname)

        @pzp.action.define(self, "Load Configuration")
        def load_config(self):
            # Open file dialog and get the selected file path
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "Select File", "",
                                                      "All Files (*);;Text Files (*.txt)", options=options)
            if filePath:
                xbars = np.load(filePath[:-len('xbars_npy')] + 'xbars.npy', allow_pickle = False)
                ybars = np.load(filePath[:-len('xbars_npy')] + 'ybars.npy', allow_pickle = False)
    
                for i in range(min(len(xbars), len(ybars))):
                    for j in range(len(xbars[0])):
                        self.combos[2*i][j].setCurrentIndex(xbars[i][j])
                    for j in range(len(ybars[0])):
                        self.combos[2*i + 1][j].setCurrentIndex(ybars[i][j])
                if len(xbars) > len(ybars):
                    for j in range(len(xbars[0])):
                        self.combos[-1][j].setCurrentIndex(xbars[-1][j])
            
    
    def custom_layout(self):
        main_layout = pg.QtWidgets.QGridLayout()
        
        w = pg.QtWidgets.QWidget()
        main_layout.addWidget(w)
        
        self.grid = layout = pg.QtWidgets.QGridLayout()
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 5)
        w.setLayout(layout)
        
        # Main image view
        self.iv1 = iv1 = pg.ImageView()
        # Hide the ROI button
        iv1.ui.roiBtn.hide()
        # Hide the Menu button
        iv1.ui.menuBtn.hide()
        layout.addWidget(iv1)
        iv1.getView().addItem(crosshair_v := pg.InfiniteLine(pos = 0, angle=90, movable=True)) #change pos to be in middle?
        iv1.getView().addItem(crosshair_h := pg.InfiniteLine(pos = 0, angle=0, movable=True))
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=True)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=True)
        self.crosshair_v = crosshair_v
        self.crosshair_h = crosshair_h

        
        # Full illumination spectrum view
        self.iv2 = iv2 = pg.ImageView()
        # Hide the ROI button
        iv2.ui.roiBtn.hide()
        # Hide the Menu button
        iv2.ui.menuBtn.hide()
        iv2.setColorMap(pg.colormap.get('afmhot', source = 'matplotlib'))
        iv2.getView().setYLink(iv1.getView())
        iv2.getView().setXLink(iv1.getView())
        layout.addWidget(iv2, 1, 0) # adds to row 1, col 0
        

        # Set up Graphics View
        self.view = view = pg.GraphicsView()
        self.scene = pg.GraphicsScene()
        view.setScene(self.scene)
        layout.addWidget(view, 0, 1, 2, 1)
        
        return main_layout

    def set_file(self, filename=None):
        """Load AFM and MFM image into the GUI"""
        if filename is None:
            filename = self.filename
        else:
            filename = filename[1:]
            self.filename = filename

        # Load images 
        self.img_AFM = np.loadtxt(filename[:-len("MFM.txt")] + 'AFM.txt')
        self.img_MFM = np.loadtxt(filename[:-len("MFM.txt")] + 'MFM.txt')

        # Move crosshairs to center of image
        self.crosshair_h.setValue(int(self.img_AFM.shape[0]/2))
        self.crosshair_v.setValue(int(self.img_AFM.shape[1]/2))

        # Rotate images
        self.rot_image()


    def rot_image(self):
        """Rotate AFM and MFM images"""
        height, width = self.img_AFM.shape[:2]
        center = (int(width/2), int(height/2))

        # Define rotation matrix
        rot_matrix = getRotationMatrix2D(center, self.params["Rotation"].value, 1)

        # Rotate images according to rotation matrix
        self.rot_AFM = warpAffine(self.img_AFM, rot_matrix,(height, width))
        self.rot_MFM = warpAffine(self.img_MFM, rot_matrix,(height, width))

        # Update images in GUI
        self.iv1.setImage(self.rot_AFM.T)
        self.iv2.setImage(self.rot_MFM.T)

    
    def updateGraphicsView(self):
        # Get the current view range from ImageView
        item = self.grid.itemAtPosition(0, 1)
        if item is not None:
            xrange = self.iv1.getView().viewRange()[0]
            yrange = self.iv1.getView().viewRange()[1]
    
            # Set the same range to GraphicsView
            self.view.setRange(QRectF(self.mult*(xrange[0] - self.origin[0]), 
                                      self.mult*(yrange[0] - self.origin[1]), 
                                      int(self.mult*(xrange[1]-xrange[0])), 
                                      int(self.mult*(yrange[1]-yrange[0]))), 
                                      padding=0)
        else:
            print("Error: No existing checkerboard")
        
    
    def populate_checkerboard(self, rows, cols, image_size, mult = 5):
        cols = 2*cols-1
        rows = 2*rows-1
        
        size_x, size_y = (image_size[0]/(cols - 1 + self.params['x_stretch'].value/10), 
                          image_size[1]/(rows - 1 + self.params['y_stretch'].value/10))
        # print('sizes', size_x, size_y)
        self.scene.clear()
        self.scene.setBackgroundBrush(QBrush(QColor(128, 128, 128)))
        self.combos = []
        for i in range(rows):
            combos = []
            for j in range(cols):
                if (i+j) % 2 == 1:
                    # Add ComboBox in a proxy widget
                    combo = QComboBox()

                    if i%2 == 0:
                        combo.addItems(['→', '←','↷','↶','U'])
                        orientation = 0
                    else:
                        combo.addItems(['↑', '↓', '↷','↶','U'])
                        orientation = 1
                    
                    combo.setFrame(False)
                    proxy = QGraphicsProxyWidget()
                    proxy.setWidget(combo)

                    pos = ((cols - j - 1)*size_x, (rows - i - 1)*size_y) # doesn't hit cols, rows
                    proxy.setPos(mult*pos[0], mult*pos[1])
                    self.scene.addItem(proxy)
                    combo.highlighted.connect(fpartial(self.highlight_MFM, pos, orientation))
                    combos.append(combo)
            self.combos.append(combos)
                    
        self.scene.setSceneRect(self.origin[0], self.origin[1], cols * iround(mult*size_x), rows * iround(mult*size_y)) # Change this for centering

        self.mult = mult
        self.iv1.getView().sigXRangeChanged.connect(self.updateGraphicsView)
        self.iv1.getView().sigYRangeChanged.connect(self.updateGraphicsView)


    def highlight_MFM(self, pos, orientation):
        # Adding a rectangle into MFM image to highlight the bar being selected in the manual input window
        self.params['Latest error'].set_value(f'highlighting working:{pos}')
        view = self.iv2.getView()
        if self.rect is not None:
            view.removeItem(self.rect)
        
        if orientation == 0:
            self.rect = pg.RectROI([self.origin[0] + pos[0] - self.bar_width/2, self.origin[1] + pos[1] - self.bar_width/8], 
                              [self.bar_width, self.bar_width/4], pen='b')
        else:
            self.rect = pg.RectROI([self.origin[0] + pos[0] - self.bar_width/8, self.origin[1]+ pos[1] - self.bar_width/2],
                              [self.bar_width/4, self.bar_width], pen='b')
        view.addItem(self.rect)


def main():
    reload(fv)
    # shell = get_ipython()
    app = QApplication(sys.argv)
    w = fv.ManyFilesViewer(ModeView)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
