# -*- coding: utf-8 -*-
"""
@author: tfarchy

Packages to install: python -m pip install puzzlepiece numpy pyqtgraph opencv-python scipy PyQt5 matplotlib
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
from PyQt5.QtWidgets import QApplication, QFileDialog, QComboBox, QGraphicsProxyWidget, QGraphicsRectItem
from PyQt5.QtGui import QBrush, QColor, QPainter
from PyQt5.QtCore import QRectF, Qt
from MFM_read_toolkit import square_lattice_bars

combo_colors = ['darkRed','darkBlue', 'darkGreen', 'darkYellow', 'black']

def save_config_helper(bars, filename):
    np.save(filename + '_xbars.npy', np.array(bars[0]), allow_pickle = False)
    np.save(filename + '_ybars.npy', np.array(bars[1]), allow_pickle = False)

def save_config_helper_csv(bars, filename):
    np.savetxt(filename + '_config.csv', bars, delimiter=',', fmt='%s')

def iround(num):
    return int(np.round(num))

def clean_peaks(data_raw, quant = 0.8, dist: int = 8):
    ns = np.arange(len(data_raw))
    fit = np.polyfit(ns, data_raw, deg = 1)
    data = data_raw - fit[0]*ns - fit[1]
    peak_i, peak_h = find_peaks(data, height = (np.quantile(data, quant), np.max(data) + 1.), distance = dist)
    return peak_i

class RectItem(QGraphicsRectItem):
    def paint(self, painter, option, widget=None):
        super(RectItem, self).paint(painter, option, widget)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.darkRed)
        painter.drawRect(option.rect)
        painter.restore()



class ScanView(fv.FileView):
    def define_params(self):
        rot = pzp.param.spinbox(self, "Rotation", 0.0)(None)
        filename = pzp.param.text(self, "Filename", 'Placeholder')(None)
        default_satx = pzp.param.spinbox(self, "Default x-bar saturation: → = 0 or ← = 1", 0)(None) 
        default_saty = pzp.param.spinbox(self, "Default y-bar saturation: ↑ = 0 or ↓ = 1", 0)(None)
        view_link = pzp.param.checkbox(self, "AFM MFM axis link", True)(None)
        mfm_boxes = pzp.param.checkbox(self, "MFM bar boxes", False)(None)
        # vertex_posx = pzp.param.spinbox(self, "Vertex pos x", 0.0)(None)
        # vertex_posy = pzp.param.spinbox(self, "Vertex pos y", 0.0)(None)
        rot.changed.connect(self.rot_image)
        view_link.changed.connect(self.image_view_link)
        mfm_boxes.changed.connect(self.show_all_bars)
        pzp.param.readout(self, 'Latest error')(None)
        # self.rect = None
        self.pre_box = None # previously seleted bar box stored here
        self.pre_combo = None
        self.mfm_view = None # MFM window ROI stored here
        self.multx, self.multy = 5, 2
        self.combos = None # QComboBoxes stored here
        self.mfm_boxes = None # MFM bar boxes stored here

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
                for i, combo_lst in enumerate(self.combos):
                    if i%2 == 0: # xbars
                        default_val = self.params["Default x-bar saturation: → = 0 or ← = 1"].value
                    else: # ybars
                        default_val = self.params["Default y-bar saturation: ↑ = 0 or ↓ = 1"].value
                    for combo in combo_lst:
                        combo.setCurrentIndex(default_val)
            else:
                self.params['Latest error'].set_value('Generate checkerboard first')
        
        @pzp.action.define(self, "Save Configuration ")
        def save_config(self):
            if len(self.combos) > 0:
                # xbars = []
                # ybars = []
                # i = 0
                # for combo_lst in self.combos:
                #     bars_lst = []
                #     for combo in combo_lst:
                #         bars_lst.append(combo.currentIndex())
                #     if i%2 == 0:
                #         xbars.append(bars_lst)
                #     else:
                #         ybars.append(bars_lst)
                #     i += 1
                empty_val = None
                all_bars = np.zeros((len(self.combos), len(self.combos[0])*2 + 1))
                for i, combo_lst in enumerate(self.combos):
                    if i%2 == 0: # xbars
                        all_bars[i, 0] = empty_val
                        for j, combo in enumerate(combo_lst):
                            all_bars[i, 2*(j+1) - 1] = combo.currentIndex()
                            all_bars[i, 2*(j+1)] = empty_val # Add empty space for vertex
                    else: # ybars
                        all_bars[i, 0] = combo_lst[0].currentIndex() + 0.5 # 0.5 to mark ybars vs xbars
                        for j, combo in enumerate(combo_lst[1:]):
                            all_bars[i, 2*(j+1) - 1] = empty_val # Empty space for vertex
                            all_bars[i, 2*(j+1)] = combo.currentIndex() + 0.5 # 0.5 to mark ybars vs xbars

            else:
                self.params['Latest error'].set_value('Generate checkerboard first')

            # if 'saved_scans' not in os.listdir(os.getcwd()):
            #     os.mkdir(os.getcwd() + '//saved_scans')
            fname = (self.params['Filename'].value +
                  f"_rot{self.params['Rotation'].value}")
            # save_config_helper((xbars, ybars), fname)
            np.savetxt(fname + '_config.csv', np.flip(np.flip(all_bars, axis=0), axis =1), delimiter=',', fmt='%s')


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
        self.graphics_view = graphics_view = pg.GraphicsView()
        self.scene = pg.GraphicsScene()
        graphics_view.setScene(self.scene)
        layout.addWidget(graphics_view, 0, 1, 2, 1)

        # Set MFM plot to change views of everything linked to it
        iv2.getView().sigXRangeChanged.connect(self.updateGraphicsView)
        iv2.getView().sigYRangeChanged.connect(self.updateGraphicsView)
        
        return main_layout

    def set_file(self, filename=None):
        """Load AFM and MFM image into the GUI"""
        if filename is None:
            filename = self.filename
        else:
            filename = filename[1:]
            self.filename = filename

        self.params['Filename'].set_value(filename[:-len("_MFM.txt")])

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
        """
        This method manages the linking of bottom left MFM plot with the view of the right checkerboard.
        Only used after the checkerboard has been initialised in function ScanView.populate_checkerboard. 
        """
        # Get the current view range from ImageView
        xrange, yrange = self.iv2.getView().viewRange()

        item = self.grid.itemAtPosition(0, 1)
        if self.combos is not None:
            # Set the same range to GraphicsView
            self.graphics_view.setRange(QRectF(self.multx*(xrange[0] - self.origin[0]), 
                                        self.multy*(yrange[0] - self.origin[1]), 
                                        int(self.multx*(xrange[1]-xrange[0])), 
                                        int(self.multy*(yrange[1]-yrange[0]))), 
                                        padding=0)
            
            if self.params['AFM MFM axis link'].value == False:
                self.updateMFMView(xrange, yrange)
        # else:
        #     print("Error: No existing checkerboard")

        if self.params['AFM MFM axis link'].value == False:
            self.updateMFMView(xrange, yrange)

    def image_view_link(self):
        """ 
        This method manages the implementation of the 'AFM MFM axis link' checkbox.
        Either axes are linked or only MFM scrolls with the MFM view shown as a rectangular ROI in the AFM plot
        """

        viewbox2 = self.iv2.getView()
        viewbox1 = self.iv1.getView() # change these
        if self.params['AFM MFM axis link'].value == True:
            viewbox2.setYLink(viewbox1)
            viewbox2.setXLink(viewbox1)
            if self.mfm_view is not None:
                viewbox1.removeItem(self.mfm_view)

        else:
            viewbox2.setYLink(None)
            viewbox2.setXLink(None)
            viewbox1.autoRange()
            xrange, yrange = viewbox2.viewRange()
            self.updateMFMView(xrange, yrange)

    def updateMFMView(self, xrange, yrange):
        """ 
        This method updates the rectangular ROI in the top left AFM plot according to the viewing range of the bottom left MFM plot.
        Only used if 'AFM MFM axis link' is set to False
        """

        viewbox1 = self.iv1.getView()
        if self.mfm_view is not None:
            viewbox1.removeItem(self.mfm_view)
        self.mfm_view = pg.RectROI([xrange[0], yrange[0]],
                                    [xrange[1] - xrange[0], yrange[1] - yrange[0]],
                                    pen='w')
        viewbox1.addItem(self.mfm_view)
        
    def populate_checkerboard(self, rows, cols, image_size):
        """
        This method populated the checkerboard on the right side of the windows and adds corresponding rectangles to bottom left MFM plot.
        """

        cols = 2*cols-1
        rows = 2*rows-1
        
        size_x, size_y = (image_size[0]/(cols - 1), 
                          image_size[1]/(rows - 1))

        self.scene.clear()
        self.scene.setBackgroundBrush(QBrush(QColor(128, 128, 128)))
        mfm_view = self.iv2.getView()

        self.combos = []
        self.mfm_boxes = []
        for i in range(rows):
            combos = []
            boxes = []
            i_ = 0
            for j in range(cols):
                j_ = 0
                # Add vertices boxes
                if i%2 == 0 and j%2 == 0:
                    pos = ((cols - j - 1)*size_x, (rows - i - 1)*size_y)
                    box = RectItem(QRectF(self.multx*pos[0], self.multy*pos[1], self.multx*size_x, self.multy*size_y))
                    self.scene.addItem(box)
                
                if (i+j) % 2 == 1:
                    # Add ComboBox in a proxy widget
                    combo = QComboBox()

                    if i%2 == 0:
                        combo.addItems(['→', '←','↷','↶','U']) # ['↑', '↓', '↷','↶','U']
                    else:
                        combo.addItems(['↑', '↓', '↷','↶','U'])
                    
                    combo.setFrame(False)
                    proxy = QGraphicsProxyWidget()
                    proxy.setWidget(combo)

                    pos = ((cols - j - 1)*size_x, (rows - i - 1)*size_y) # doesn't hit cols, rows
                    proxy.setPos(self.multx*pos[0], self.multy*pos[1])
                    self.scene.addItem(proxy)

                    if i%2 == 0:
                        box = pg.RectROI([self.origin[0] + pos[0] - self.bar_width/2, self.origin[1] + pos[1] - self.bar_width/8],
                                               [self.bar_width, self.bar_width/4],
                                               pen='b')
                    else:
                        box = pg.RectROI([self.origin[0] + pos[0] - self.bar_width/8, self.origin[1]+ pos[1] - self.bar_width/2],
                                               [self.bar_width/4, self.bar_width],
                                               pen='b')
                    box.setVisible(False)
                    box.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
                    mfm_view.addItem(box)

                    index = (i_, j_)
                    box.sigClicked.connect(fpartial(self.highlight_combo, combo)) # hover event too?
                    combo.highlighted.connect(fpartial(self.highlight_mfm_box, box))
                    combos.append([combo, pos])
                    boxes.append(box)
                    j_ += 1
            i_ += 1
            self.combos.append(combos)
            self.mfm_boxes.append(boxes)
                    
    def show_all_bars(self):
        """ 
        Manages the implementation of the 'MFM bar boxes' checkbox.
        Sets all bar boxes in bottom left MFM plot to visible and resets their colour to default (blue).
        """
        if self.mfm_boxes is not None:
            for box in [box for box_lst in self.mfm_boxes for box in box_lst]:
                box.setPen('b')
                box.setVisible(self.params['MFM bar boxes'].value)
        else:
            self.params['Latest error'].set_value("Checkerboard not initialised")

    def highlight_mfm_box(self, box):
        """
        Manages the connection between a QComboBox being selected on the right checkerboard
        and the corresponding bar being highlighted in the bottom left MFM plot.
        Only used once the checkerboard has been initialised.
        """

        if self.pre_box is not None:
            self.pre_box.setPen('r')
        self.pre_box = box

        # Set new box to visible and change colour
        box.setPen('g')
        box.setVisible(True)

    def highlight_combo(self, combo):
        """
        Manages the connection between a box being selected in the bottom left MFM plot
        and the corresponding QComboBox being highlighted on the right checkerboard.
        Only used once the checkerboard has been initialised.
        """
        if self.pre_combo is not None:
            self.pre_combo.hidePopup()
        self.pre_combo = combo
        combo.showPopup()



def main():
    reload(fv)
    # shell = get_ipython()
    app = QApplication(sys.argv)
    w = fv.ManyFilesViewer(ScanView)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
