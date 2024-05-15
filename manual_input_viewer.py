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

class CustomTextItem(pg.TextItem):
    sigClicked = pg.Qt.QtCore.Signal(object)

    def __init__(self, text="", **kwargs):
        super(CustomTextItem, self).__init__(text, **kwargs)
        self.setAcceptHoverEvents(True)
        self._text = text  # Store the text

    def getText(self):
        return self._text  # Method to retrieve the text
    
    def customSetText(self, text):
        self._text = text
        super(CustomTextItem, self).setText(text)
    
    def mousePressEvent(self, event):
        self.sigClicked.emit(self)  # Emit the signal passing the text item instance
        super(CustomTextItem, self).mousePressEvent(event)  # Continue processing the parent class event
        event.accept()


class ScanView(fv.FileView):
    def define_params(self):
        rot = pzp.param.spinbox(self, "Rotation", 0.0)(None)
        filename = pzp.param.text(self, "Filename", 'Placeholder')(None)
        default_satx = pzp.param.spinbox(self, "Default x-bar saturation: ← = 1 or → = 2", 0)(None) 
        default_saty = pzp.param.spinbox(self, "Default y-bar saturation: ↑ = 1 or ↓ = 2", 0)(None)
        size = pzp.param.spinbox(self, "Arrow point size", 0)(None)
        view_link = pzp.param.checkbox(self, "AFM MFM axis link", True)(None)
        mfm_boxes = pzp.param.checkbox(self, "MFM bar boxes", True)(None)
        rot.changed.connect(self.rot_image)
        size.changed.connect(self.arrow_size)
        view_link.changed.connect(self.image_view_link)
        mfm_boxes.changed.connect(self.show_all_bars)
        pzp.param.readout(self, 'Latest error')(None)
        self.mfm_view = None # MFM window ROI stored here
        self.multx, self.multy = 5, 2
        self.mfm_boxes = None # MFM bar boxes stored here
        self.text_items = []
        self.h_options = ['U', '←', '→']
        self.v_options = ['U', '↑', '↓']
        self.colors = ['r', 'b', 'g']

    def define_actions(self):
        @pzp.action.define(self, "Generate ASI Input")
        def generate_checkerboard(self):
            # Read horizontal and vertical lines
            y, x = int(np.round(self.line_h.value())), int(np.round(self.line_v.value()))
    
            # Predict SI dimensions
            vrtx_coords, vrtx_img, (xbars_img, ybars_img) = square_lattice_bars(self.rot_AFM, self.rot_MFM, x, y)
            self.ASI_shape = vrtx_coords.shape[:1]
            h_bar_width = (vrtx_coords[1, 0, 1] - vrtx_coords[0, 0, 1])/4
            v_bar_width = (vrtx_coords[0, 1, 0] - vrtx_coords[0, 0, 0])/4

            x_boxes = []
            y_boxes = []
            x_bars = []
            y_bars = []
            for row in range(len(vrtx_coords)):
                row_x_boxes = []
                row_y_boxes = []
                row_x_bars = []
                row_y_bars = []
                for col in range(len(vrtx_coords[0])):
                    mfm_view = self.iv2.getView()
                    # Adding boxes to mfm view
                    if col < len(vrtx_coords[0]) - 1:

                        # Build the ROI box
                        h_box = pg.RectROI([vrtx_coords[row, col, 0], (vrtx_coords[row, col, 1] + vrtx_coords[row, col + 1, 1])/2 - h_bar_width/2],
                                           [vrtx_coords[row, col + 1, 0] - vrtx_coords[row, col, 0], h_bar_width],
                                            pen = pg.mkPen(None),
                                            movable = False)
                        
                        # Build the bar
                        state = self.pred_mag(xbars_img[row, col])
                        h_state = CustomTextItem(text = self.h_options[state],
                                                 anchor = (0.5, 0.5),
                                                 color = self.colors[state])
                        h_state.setPos((vrtx_coords[row, col, 0] + vrtx_coords[row, col + 1, 0])/2, (vrtx_coords[row, col, 1] + vrtx_coords[row, col + 1, 1])/2)
                        font = h_state.textItem.font()
                        font.setBold(True)

                        mfm_view.addItem(h_state)
                        mfm_view.addItem(h_box)

                        for handle in h_box.getHandles():
                            h_box.removeHandle(handle)

                        h_state.sigClicked.connect(fpartial(self.h_switch_state, h_state, h_box))

                        row_x_boxes.append(h_box)
                        row_x_bars.append(h_state)

                    if row < len(vrtx_coords) - 1:
                        v_box = pg.RectROI([(vrtx_coords[row, col, 0] + vrtx_coords[row + 1, col, 0])/2 - v_bar_width/2, vrtx_coords[row, col, 1]],
                                           [v_bar_width, vrtx_coords[row + 1, col, 1] - vrtx_coords[row, col, 1]],
                                           pen = pg.mkPen(None),
                                           movable = False)
                        
                        # Build the bar
                        state = self.pred_mag(ybars_img[row, col])
                        v_state = CustomTextItem(text = self.v_options[state],
                                                anchor = (0.5, 0.5),
                                                color = self.colors[state])
                        v_state.setPos((vrtx_coords[row, col, 0] + vrtx_coords[row + 1, col, 0])/2, (vrtx_coords[row, col, 1] + vrtx_coords[row + 1, col, 1])/2)
                        font = v_state.textItem.font()
                        font.setBold(True)

                        mfm_view.addItem(v_state)
                        mfm_view.addItem(v_box)
                        for handle in v_box.getHandles():
                            v_box.removeHandle(handle)

                        v_state.sigClicked.connect(fpartial(self.v_switch_state, v_state, v_box))

                        row_y_boxes.append(v_box)
                        row_y_bars.append(v_state)

                x_boxes.append(row_x_boxes)
                x_bars.append(row_x_bars)
                if len(row_y_bars) != 0: # Exclude the last row where this will be empty
                    y_boxes.append(row_y_boxes)
                    y_bars.append(row_y_bars)
            self.mfm_boxes = [x_boxes, y_boxes]
            self.text_items = [x_bars, y_bars]
            self.mfm_ref = mfm_view.viewRange()[0][1] - mfm_view.viewRange()[0][0]
            self.params['Arrow point size'].set_value(self.text_items[0][0][0].textItem.font().pointSize())

        @pzp.action.define(self, "Default mag")
        def default_checkerboard(self):
            if len(self.text_items) > 0:
                for text in [text_item for text_lst in self.text_items[0] for text_item in text_lst]:
                    text.customSetText(self.h_options[self.params["Default x-bar saturation: ← = 1 or → = 2"]])
                for text in [text_item for text_lst in self.text_items[1] for text_item in text_lst]:
                    text.customSetText(self.v_options[self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"]])
            else:
                self.params['Latest error'].set_value('Generate checkerboard first')
        
        @pzp.action.define(self, "Save Configuration ")
        def save_config(self):
            if len(self.text_items) > 0:
                empty_val = None
                all_bars = np.zeros((len(self.text_items[0]) + len(self.text_items[1]),
                                     len(self.text_items[0][0]) + len(self.text_items[1][0])))
                for row in range(len(self.text_items[0]) + len(self.text_items[1])):
                    if row%2 == 0: # xbars
                        all_bars[row, 0] = empty_val
                        for j, text in enumerate(self.text_items[0][int(row/2)]):
                            all_bars[row, 2*(j+1) - 1] = self.h_options.index(text.getText())
                            all_bars[row, 2*(j+1)] = empty_val # Add empty space for vertex
                    else: # ybars
                        items = self.text_items[1][int(row/2)]
                        all_bars[row, 0] = self.v_options.index(items[0].getText()) + 0.5 # 0.5 to mark ybars vs xbars
                        for j, text in enumerate(self.text_items[1][int(row/2)][1:]):
                            all_bars[row, 2*(j+1) - 1] = empty_val # Empty space for vertex
                            all_bars[row, 2*(j+1)] = self.v_options.index(text.getText()) + 0.5 # 0.5 to mark ybars vs xbars

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

        main_layout.setRowStretch(0, 3)
        # main_layout.setRowStretch(1, 3)
        
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
        layout.addWidget(iv1, 0, 0, 2, 1)
        line_v = pg.InfiniteLine(pos = 0, angle=90, movable=True)
        line_h = pg.InfiniteLine(pos = 0, angle=0, movable=True)
        iv1.getView().addItem(line_v) #change pos to be in middle?
        iv1.getView().addItem(line_h)
        self.line_v = pg.InfiniteLine(angle=90, movable=True)
        self.line_h = pg.InfiniteLine(angle=0, movable=True)
        self.line_v = line_v
        self.line_h = line_h

        
        # Full illumination spectrum view
        self.iv2 = iv2 = pg.ImageView()
        # Hide the ROI button
        iv2.ui.roiBtn.hide()
        # Hide the Menu button
        iv2.ui.menuBtn.hide()
        iv2.setColorMap(pg.colormap.get('afmhot', source = 'matplotlib'))
        iv2.getView().setYLink(iv1.getView())
        iv2.getView().setXLink(iv1.getView())
        layout.addWidget(iv2, 0, 1, 2, 1) # adds to row 1, col 0
                
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
        self.line_h.setValue(int(self.img_AFM.shape[0]/2))
        self.line_v.setValue(int(self.img_AFM.shape[1]/2))

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
            self.updateMFMView()
            self.iv2.getView().sigRangeChanged.connect(self.updateMFMView)

    def updateMFMView(self):
        """ 
        This method updates the rectangular ROI in the top left AFM plot according to the viewing range of the bottom left MFM plot.
        Only used if 'AFM MFM axis link' is set to False
        """
        xrange, yrange = self.iv2.getView().viewRange()
        viewbox1 = self.iv1.getView()
        if self.mfm_view is not None:
            viewbox1.removeItem(self.mfm_view)
        self.mfm_view = pg.RectROI([xrange[0], yrange[0]],
                                    [xrange[1] - xrange[0], yrange[1] - yrange[0]],
                                    pen='w')
        viewbox1.addItem(self.mfm_view)

    def show_all_bars(self):
        """ 
        Manages the implementation of the 'MFM bar boxes' checkbox.
        Sets all bar boxes in bottom left MFM plot to visible and resets their colour to default (blue).
        """
        if self.mfm_boxes is not None:
            for box in [box for box_lst in self.mfm_boxes for box_row in box_lst for box in box_row]:
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

    def pred_mag(self, img):
        if img.shape[0] > img.shape[1]:
            img = img.T
        sect = [np.mean(array) for array in np.array_split(np.sum(img[1:-2], axis = 0), 4)]

        if np.argmax(np.abs(sect)) != 0 and np.argmax(np.abs(sect)) != 3:
            return 0
        elif sect[0] > sect[-1]:
            return 1
        else:
            return 2
        
    def v_switch_state(self, text, box):
        state_index = self.v_options.index(text.getText())
        text.customSetText(self.v_options[(state_index + 1) % len(self.v_options)])
        text.setColor(self.colors[(state_index + 1) % len(self.v_options)])
        text.update()
        box.setPen('darkRed')

    def h_switch_state(self, text, box):
        state_index = self.h_options.index(text.getText())
        new_index = (state_index + 1) % len(self.h_options)
        text.customSetText(self.h_options[new_index])
        text.setColor(self.colors[new_index])
        text.update()
        box.setPen('darkRed')

    def arrow_size(self):
        for text_item in [item for item_cat in self.text_items for item_row in item_cat for item in item_row]:
            font = text_item.textItem.font()
            font.setPointSize(self.params['Arrow point size'].value)  # Adjust scaling factor as needed
            text_item.setFont(font)







def main():
    reload(fv)
    # shell = get_ipython()
    app = QApplication(sys.argv)
    w = fv.ManyFilesViewer(ScanView)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
