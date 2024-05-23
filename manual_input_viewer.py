# -*- coding: utf-8 -*-
"""
@author: tfarchy

Packages to install: python -m pip install puzzlepiece numpy pyqtgraph opencv-python scipy PyQt5 matplotlib imageio scipy
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
from cv2 import getRotationMatrix2D, warpAffine, resize, INTER_CUBIC
from scipy.signal import find_peaks
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsEllipseItem, QDoubleSpinBox, QTreeWidgetItem, QTreeWidget, QPushButton, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QCheckBox
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtCore import QRectF, Qt
from MFM_read_toolkit import square_lattice_bars
import imageio.v3 as iio
from scipy.ndimage import zoom, affine_transform
import matplotlib.pyplot as plt


vertex_colors = ['red', 'green', 'dodgerblue', 'yellow', 'orange', 'purple', 'darkblue']

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

def rotation_helper(img, rot):
    height, width = img.shape[:2]
    center = (int(width/2), int(height/2))

    # Define rotation matrix
    rot_matrix = getRotationMatrix2D(center, rot, 1)

    # Rotate images according to rotation matrix
    rot_img = warpAffine(img, rot_matrix, dsize = (height, width))

    return rot_img

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

class CustomCheckBox(QCheckBox):
    def __init__(self, init_value: bool = False):
        super(QCheckBox, self).__init__()
        self.setValue(init_value)
    
    def setValue(self, value):
        self.setCheckState(value)
        self.value = value

class ScanView(fv.FileView):
    def define_params(self):
        rot = pzp.param.spinbox(self, "Rotation", 0.0)(None)
        filename = pzp.param.text(self, "Filename", 'Placeholder')(None)
        default_satx = pzp.param.spinbox(self, "Default x-bar saturation: ← = 1 or → = 2", 0)(None) 
        default_saty = pzp.param.spinbox(self, "Default y-bar saturation: ↑ = 1 or ↓ = 2", 0)(None)
        size = pzp.param.spinbox(self, "Arrow point size", 0)(None)
        vertices = pzp.param.checkbox(self, "Show vertices", False)(None)
        arrows = pzp.param.checkbox(self, "Show arrows", False)(None)
        mfm_boxes = pzp.param.checkbox(self, "Highlight arrow changes", True)(None)
        rot.changed.connect(self.rot_image)
        size.changed.connect(self.arrow_size)
        mfm_boxes.changed.connect(self.show_all_bars)
        vertices.changed.connect(self.show_vertices)
        arrows.changed.connect(self.show_arrows)

        pzp.param.readout(self, 'Latest error')(None)
        self.mfm_view = None # MFM window ROI stored here
        self.multx, self.multy = 5, 2
        self.mfm_boxes = None # MFM bar boxes stored here
        self.vertices = None
        self.mask = None
        self.mask_item = None
        self.text_items = []
        self.h_options = ['U', '←', '→']
        self.v_options = ['U', '↑', '↓']
        self.colors = ['r', 'b', 'g']
        self.vrtx_options = ['U', 'T1', 'T2_og', 'T2', 'T3.1', 'T3.2', 'T4.1', 'T4.1']

    def define_actions(self):
        @pzp.action.define(self, "Generate ASI Input")
        def generate_checkerboard(self):
            # Read horizontal and vertical lines
            y, x = int(np.round(self.line_h.value())), int(np.round(self.line_v.value()))
    
            # Predict SI dimensions
            vrtx_coords, vrtx_img, (xbars_img, ybars_img) = square_lattice_bars(self.rot_AFM, self.rot_MFM, x, y)
            self.vrtx_coords = vrtx_coords
            self.ASI_shape = vrtx_coords.shape[:1]
            h_bar_width = (vrtx_coords[1, 0, 1] - vrtx_coords[0, 0, 1])/3
            v_bar_width = (vrtx_coords[0, 1, 0] - vrtx_coords[0, 0, 0])/3
            self.bar_width = (h_bar_width + h_bar_width)/2

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
                                            pen = 'cyan',
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
                        h_box.setVisible(False)
                        row_x_bars.append(h_state)

                    if row < len(vrtx_coords) - 1:
                        v_box = pg.RectROI([(vrtx_coords[row, col, 0] + vrtx_coords[row + 1, col, 0])/2 - v_bar_width/2, vrtx_coords[row, col, 1]],
                                           [v_bar_width, vrtx_coords[row + 1, col, 1] - vrtx_coords[row, col, 1]],
                                           pen = 'cyan',
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
                        v_box.setVisible(False)
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
                    text.customSetText(self.h_options[self.params["Default x-bar saturation: ← = 1 or → = 2"].value])
                    text.setColor(self.colors[self.params["Default x-bar saturation: ← = 1 or → = 2"].value])
                for text in [text_item for text_lst in self.text_items[1] for text_item in text_lst]:
                    text.customSetText(self.v_options[self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value])
                    text.setColor(self.colors[self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value])
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

                fname = (self.params['Filename'].value +
                  f"_rot{self.params['Rotation'].value}")
                np.savetxt(fname + '_config.csv', np.flip(np.flip(all_bars, axis=0), axis =1), delimiter=',', fmt='%s')

            else:
                self.params['Latest error'].set_value('Generate checkerboard first')

            # if 'saved_scans' not in os.listdir(os.getcwd()):
            #     os.mkdir(os.getcwd() + '//saved_scans')

        @pzp.action.define(self, "Load Configuration")
        def load_config(self):
            # Open file dialog and get the selected file path
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getOpenFileName(self, "Select File", "",
                                                      "All Files (*);;Text Files (*.txt)", options=options)
            if filePath:
                checkerboard = np.flip(np.flip(np.loadtxt(filePath, delimiter = ','), axis = 1), axis = 0)
                for i, row in enumerate(checkerboard):
                    if i % 2 == 0: # xbars
                        row_states = [int(state_index) for state_index in [elem for elem in row if elem in np.arange(len(self.h_options))]]
                        # print(row_states)
                        for text_item, state_index in zip(self.text_items[0][int(i/2)], row_states):
                            text_item.customSetText(self.h_options[state_index])
                            text_item.setColor(self.colors[state_index])
                    else: #ybars
                        row_states = [int(state_index - 0.5) for state_index in [elem for elem in row if elem - 0.5 in np.arange(len(self.v_options))]]
                        for text_item, state_index in zip(self.text_items[1][int(i/2)], row_states):
                            text_item.customSetText(self.v_options[state_index])
                            text_item.setColor(self.colors[state_index])
            
            
        @pzp.action.define(self, "Evaluate vertices")
        def evaluate_vertices(self):
            self.build_vertices()
            
    def custom_layout(self):
        main_layout = pg.QtWidgets.QGridLayout() # main_layout = overall layout of whole window
        
        w = pg.QtWidgets.QWidget()
        main_layout.addWidget(w)

        # main_layout.setRowStretch(0, 2)
        # main_layout.setRowStretch(1, 2)
        
        self.grid = layout = pg.QtWidgets.QGridLayout() # layout = sublayout excluding buttons at the top
        # layout.setColumnStretch(0, 6)
        # layout.setColumnStretch(1, 2)
        # layout.setRowStretch(0, 1)
        # layout.setRowStretch(1, 10)
        w.setLayout(layout)
        
        # Main image view
        self.iv1 = iv1 = pg.ImageView()
        # Hide the ROI button
        iv1.ui.roiBtn.hide()
        # Hide the Menu button
        iv1.ui.menuBtn.hide()
        layout.addWidget(iv1, 0, 0, 1, 1)
        line_v = pg.InfiniteLine(pos = 0, angle=90, movable=True)
        line_h = pg.InfiniteLine(pos = 0, angle=0, movable=True)
        crosshair = pg.TargetItem(symbol = '+', pen='b', movable=False)
        iv1.getView().addItem(line_v) #change pos to be in middle?
        iv1.getView().addItem(line_h)
        iv1.getView().addItem(crosshair)
        self.line_v = pg.InfiniteLine(angle=90, movable=True)
        self.line_h = pg.InfiniteLine(angle=0, movable=True)
        self.line_v = line_v
        self.line_h = line_h
        self.crosshair = crosshair

        
        # Full illumination spectrum view
        self.iv2 = iv2 = pg.ImageView()
        # Hide the ROI button
        iv2.ui.roiBtn.hide()
        # Hide the Menu button
        iv2.ui.menuBtn.hide()
        iv2.setColorMap(pg.colormap.get('afmhot', source = 'matplotlib'))
        iv2.getView().setYLink(iv1.getView())
        iv2.getView().setXLink(iv1.getView())
        layout.addWidget(iv2, 0, 1, 1, 1) # adds to row 1, col 0

        #Create mask overlay title for row 1 col 0
        title_row = QHBoxLayout()
        mask_title = QLabel("Mask Overlay and Adjustment")
        title_row.addWidget(mask_title, alignment = Qt.AlignCenter)
        # mask_title.setAlignment(Qt.AlignCenter)

        # Create mask button
        mask_button = QPushButton("Overlay Mask")
        mask_button.clicked.connect(self.overlay_mask)

        # Create show mask checkbox
        mask_check = CustomCheckBox()
        mask_check.setChecked(False)
        check_label = QLabel("Show Mask")
        check_row = QHBoxLayout()
        check_row.addWidget(check_label)
        check_row.addWidget(mask_check)

        # Connections to rest of code
        self.params["Show Mask"] = mask_check
        mask_check.stateChanged.connect(self.show_mask)

        #Create a layout for title, button and spinboxes
        mask_layout = QVBoxLayout()
        mask_layout.addLayout(title_row)
        mask_layout.addWidget(mask_button)
        mask_layout.addLayout(check_row)

        # Create float number input boxes
        labels = ['Mask x-stretch', 'Mask y-stretch', 'Mask x-position', 'Mask y-position']
        inits = [1.1, 1.36, 0.10, 0.32]
        bounds_lst = [(0.5, 1.5), (0.5, 1.5), (-1.0, 1.0), (-1.0, 1.0)]
        
        for key, init, bounds in zip(labels, inits, bounds_lst):
            # Create label
            label = QLabel(key)

            #Create input box
            input = QDoubleSpinBox(value = init, maximum = bounds[1], minimum = bounds[0], singleStep = 0.01, decimals = 2)

            # Create horizontal layout for each row
            row_layout = QHBoxLayout()
            row_layout.addWidget(label)
            row_layout.addWidget(input)

            # Add row layout to vertical mask layout
            mask_layout.addLayout(row_layout)

            # Manage connections to rest of the code
            input.valueChanged.connect(self.update_mask)
            self.params[key] = input
        
        # Add button to evaluate statistics
        statistics_button = QPushButton("Evaluate Statistics")
        statistics_button.clicked.connect(self.statistics_output)
        mask_layout.addWidget(statistics_button)

        # Add mask layout to grid sublayout
        layout.addLayout(mask_layout, 1, 0)
        
        
        self.iv3 = iv3 = pg.ImageView()
        # Hide the ROI button
        iv3.ui.roiBtn.hide()
        # Hide the Menu button
        iv3.ui.menuBtn.hide()
        iv3.setColorMap(pg.colormap.get('afmhot', source = 'matplotlib'))
        # iv3.getView().setYLink(iv1.getView())
        # iv3.getView().setXLink(iv1.getView())
        layout.addWidget(iv3, 1, 1, 1, 1) # adds to row 1, col 0
        
        # # Connect all mask controls
        # for mask_input in mask_inputs:
        #     mask_input.changed.connect(self.transform_mask)

        # Connect clone target
        iv2.scene.sigMouseMoved.connect(self.move_clone_target)
        iv3.scene.sigMouseMoved.connect(self.move_clone_target)

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
        # height, width = self.img_AFM.shape[:2]
        # center = (int(width/2), int(height/2))

        # # Define rotation matrix
        # rot_matrix = getRotationMatrix2D(center, self.params["Rotation"].value, 1)

        # # Rotate images according to rotation matrix
        # self.rot_AFM = warpAffine(self.img_AFM, rot_matrix,(height, width))
        # self.rot_MFM = warpAffine(self.img_MFM, rot_matrix,(height, width))

        self.rot_AFM = rotation_helper(self.img_AFM, self.params["Rotation"].value)
        self.rot_MFM = rotation_helper(self.img_MFM, self.params["Rotation"].value)

        # Update images in GUI
        self.iv1.setImage(self.rot_AFM.T)
        self.iv2.setImage(self.rot_MFM.T)
        # self.iv3.setImage(self.rot_MFM.T)
        image3 = pg.ImageItem(self.rot_MFM.T)
        self.iv3.addItem(image3)

    def overlay_mask(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select File", "",
                                                      "All Files (*);;Text Files (*.txt)", options=options)
        if filePath:
            # Remove previous mask if one exists
            if self.mask is not None:
                self.iv3.removeItem(self.mask_item)
            
            # Load new mask
            mask = iio.imread(filePath)

            # Correct for DMD stretching
            zoom_factors = (1, 0.5, 1) if mask.ndim == 3 else (1, 0.5)
            self.mask = zoom(mask.T, zoom_factors, order = 3)[450:690, 336:576] #For 60X: [96:816, 210:930]

            # Add mask to image view 3
            self.update_mask()

            # Make sure the Show Mask checkbox is set to True
            self.params["Show Mask"].setValue(True) 

    def _transform_mask(self):
        """
        Transforms the given 2D numpy array by shifting and stretching it.
        """
        # Get the shape of the original image
        original_shape = self.mask.shape
        
        # Create the transformation matrix for scaling (stretching)
        # scale_matrix = np.array([
        #     [1/self.params['Mask x-stretch'].value, 0, 0],
        #     [0, 1/self.params['Mask y-stretch'].value, 0],
        #     [0, 0, 1]
        # ])
        # print(type(self.params['Mask x-stretch'].value()), self.params['Mask x-stretch'].value())
        scale_matrix = np.array([[1/self.params['Mask x-stretch'].value(), 0],
                                [0, 1/self.params['Mask y-stretch'].value()]])

        # Create the offset for shifting
        offset = np.array([self.params['Mask x-position'].value() * 0.5 * original_shape[0],
                        self.params['Mask y-position'].value() * 0.5 * original_shape[1]])
        
        # Apply the affine transformation
        transformed_mask = affine_transform(self.mask, scale_matrix[:2, :2], offset=offset, output_shape=original_shape, order=1, mode='constant', cval=0.0)
        
        # Crop the image to the original size (if necessary)
        # transformed_mask = transformed_image[:original_shape[0], :original_shape[1]]
        rot_mask = rotation_helper(transformed_mask, self.params["Rotation"].value)

        scaled_rot_mask = resize(rot_mask, dsize = self.rot_MFM.shape[:2], interpolation = INTER_CUBIC)

        return scaled_rot_mask
        
        # self.iv3.removeItem(self.mask_item)
        # self.mask_item = pg.ImageItem(self.transformed_mask)
        # self.mask_item.setOpacity(0.5)
        # self.iv3.setImage(self.mask_item)

    def update_mask(self):
        if self.mask is not None:
            self.transformed_mask = self._transform_mask()

            # Remove existing mask
            if self.mask_item is not None:
                self.iv3.removeItem(self.mask_item)

            # Add new mask
            self.mask_item = pg.ImageItem(self.transformed_mask)
            self.mask_item.setOpacity(0.5)
            self.iv3.addItem(self.mask_item) # ERRORS HERE 

            # NOTE: easier/simpler way to do this?

        else:
            self.params['Latest error'].set_value("Initialise mask first")
    
    def show_mask(self):
        """
        Manages toggling between displaying and hiding mask in bottom right AFM image.
        If this is the first time calling the function (ie if self.mask is None), opens a file dialog window to load mask.
        """
        if self.mask is not None:
            if self.params["Show Mask"].value:
                self.mask_item.setOpacity(0.5)
            else:
                self.mask_item.setOpacity(0)
        else:
            self.params["Latest error"].set_value("Initialise mask first")

    def updateMFMView(self):
        """ 
        This method updates the rectangular ROI in the top left AFM plot according to the viewing range of the bottom left MFM plot.
        Only used if 'AFM/MFM axis link' is set to False
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
                box.setVisible(self.params['Highlight arrow changes'].value)
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
        box.setVisible(True)

    def h_switch_state(self, text, box):
        state_index = self.h_options.index(text.getText())
        new_index = (state_index + 1) % len(self.h_options)
        text.customSetText(self.h_options[new_index])
        text.setColor(self.colors[new_index])
        box.setVisible(True)

    def arrow_size(self):
        for text_item in [item for item_cat in self.text_items for item_row in item_cat for item in item_row]:
            font = text_item.textItem.font()
            font.setPointSize(self.params['Arrow point size'].value)  # Adjust scaling factor as needed
            text_item.setFont(font)

    def move_clone_target(self, pos):
        """
        Manages the synchronisation between mouse movement in the MFM ImageView and the crosshair movement in the AFM ImageView.
        """

        # Map positon of mouse to imageview1
        mapped_pos = self.iv1.getImageItem().mapFromScene(pos)

        # Update position of target in imageview1
        self.crosshair.setPos(mapped_pos)

    def build_vertices(self):
        self.vertices = []
        mfm_view = self.iv2.getView()
        for row in range(len(self.text_items[1]) - 1): # number of inner vertex rows is len(ybars) - 1
            row_vertices = []
            for col in range(len(self.text_items[0][0]) - 1): # number of innver vertex cols is len(xbars[0]) - 1
                left, right = self.text_items[0][row + 1][col].getText(), self.text_items[0][row + 1][col + 1].getText()
                top, bottom = self.text_items[1][row][col + 1].getText(), self.text_items[1][row + 1][col + 1].getText()

                xbar_macro_check = [bar_mag in ['←', '→'] for bar_mag in [left, right]]
                ybar_macro_check = [bar_mag in ['↑', '↓'] for bar_mag in [top, bottom]]

                circle = QGraphicsEllipseItem(float(self.vrtx_coords[row + 1][col + 1][0]),
                                              float(self.vrtx_coords[row + 1][col + 1][1]),
                                              self.bar_width,
                                              self.bar_width)
                
                if not all(xbar_macro_check + ybar_macro_check):
                    circle.setPen(QPen(QColor(vertex_colors[0])))
                    circle.setBrush(QBrush(QColor(vertex_colors[0])))
                    row_vertices.append((self.vrtx_options[0], circle))

                elif ([left, right, top, bottom] == ['←', '→', '↓', '↑'] or
                      [left, right, top, bottom] == ['→', '←', '↑', '↓']):
                    circle.setPen(QPen(QColor(vertex_colors[1])))
                    circle.setBrush(QBrush(QColor(vertex_colors[1])))
                    row_vertices.append((self.vrtx_options[1], circle))


                elif ([left, right, top, bottom] == ['→', '→', '↑', '↑'] or
                      [left, right, top, bottom] == ['→', '→', '↓', '↓'] or
                      [left, right, top, bottom] == ['←', '←', '↑', '↑'] or
                      [left, right, top, bottom] == ['←', '←', '↓', '↓']):
                    circle.setPen(QPen(QColor(vertex_colors[2])))
                    circle.setBrush(QBrush(QColor(vertex_colors[2])))
                    # Testing if configuration is the same as before writing
                    if [left, right, top, bottom] == [self.h_options[self.params["Default x-bar saturation: ← = 1 or → = 2"].value],
                                                      self.h_options[self.params["Default x-bar saturation: ← = 1 or → = 2"].value],
                                                      self.v_options[self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value],
                                                      self.v_options[self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value]]:
                        row_vertices.append((self.vrtx_options[2], circle))
                    else:
                        if (self.params["Default x-bar saturation: ← = 1 or → = 2"].value == 0 or
                            self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value == 0):
                            self.params["Latest error"].set_value("No default saturation. Statistics will be incomplete.")
                        row_vertices.append((self.vrtx_options[3], circle))

                # 1-in 3-out
                elif ([left, right, top, bottom] == ['←', '→', '↓', '↓'] or
                      [left, right, top, bottom] == ['←', '←', '↑', '↓'] or
                      [left, right, top, bottom] == ['←', '→', '↑', '↑'] or
                      [left, right, top, bottom] == ['→', '→', '↑', '↓']):
                    circle.setPen(QPen(QColor(vertex_colors[3])))
                    circle.setBrush(QBrush(QColor(vertex_colors[3])))
                    row_vertices.append((self.vrtx_options[4], circle))

                # 3-in 1-out
                elif ([left, right, top, bottom] == ['→', '←', '↑', '↑'] or
                      [left, right, top, bottom] == ['→', '→', '↓', '↑'] or
                      [left, right, top, bottom] == ['→', '←', '↓', '↓'] or
                      [left, right, top, bottom] == ['←', '←', '↓', '↑']):
                    circle.setPen(QPen(QColor(vertex_colors[4])))
                    circle.setBrush(QBrush(QColor(vertex_colors[4])))
                    row_vertices.append((self.vrtx_options[5], circle))

                # all-out
                elif [left, right, top, bottom] == ['←', '→', '↑', '↓']:
                    circle.setPen(QPen(QColor(vertex_colors[5])))
                    circle.setBrush(QBrush(QColor(vertex_colors[5])))
                    row_vertices.append((self.vrtx_options[6], circle))

                # all-in
                elif [left, right, top, bottom] == ['←', '→', '↑', '↓']:
                    circle.setPen(QPen(QColor(vertex_colors[6])))
                    circle.setBrush(QBrush(QColor(vertex_colors[6])))
                    row_vertices.append((self.vrtx_options[7], circle))
                
                mfm_view.addItem(circle)
            self.vertices.append(row_vertices)

    def show_vertices(self):
        """
        Allows user to toggle visibility of vertex markers once they have been evaluated.
        """
        if self.vertices is not None:
            for circle in [vertex[1] for vertex_row in self.vertices for vertex in vertex_row]:
                circle.setVisible(self.params['Show vertices'].value)
        else:
            # self.params['Show vertices'].set_value(False) # Check if this won't create infinite loop
            self.params['Latest error'].set_value('Evaluate vertices first')

    def show_arrows(self):
        """
        Allows user to toggle visibility of arrow markers once they have been generated.
        """
        if len(self.text_items) != 0:
            for text in [text_item for bar_cat in self.text_items for bar_row in bar_cat for text_item in bar_row]:
                text.setVisible(self.params['Show arrows'].value)
        else:
            # self.params['Show arrows'].set_value(False)
            self.params['Latest error'].set_value('Generate arrows first')

    def statistics_output(self):
        if self.vertices is not None and self.mask is not None:
            ## VERTICES
            written_vertices = []
            for row in range(len(self.text_items[1]) - 1): # number of inner vertex rows is len(ybars) - 1
                for col in range(len(self.text_items[0][0]) - 1): # number of innver vertex cols is len(xbars[0]) - 1
                    # Vertex position
                    x, y = self.vrtx_coords[row + 1][col + 1][0], self.vrtx_coords[row + 1][col + 1][1]

                    # If mask is dark in this area, save state for statistics
                    if self.transformed_mask[x, y] < 127: # Flipped x and y here
                        written_vertices.append(self.vertices[row][col][0])
            
            ## X ARROWS
            written_x_arrows = []
            x_arrows = self.text_items[0]
            for row, row_arrows in enumerate(x_arrows):
                for col, arrow in enumerate(row_arrows):
                    # Bar middle position
                    x, y = (int((self.vrtx_coords[row][col][0] + self.vrtx_coords[row][col + 1][0])/2),
                            int((self.vrtx_coords[row][col][1] + self.vrtx_coords[row][col + 1][1])/2))
                    
                    # If mask is dark in this area, save state for statistics
                    if self.transformed_mask[x, y] < 127: # Flipped x and y here
                        written_x_arrows.append(arrow.getText())

            ## Y ARROWS
            written_y_arrows = []
            y_arrows = self.text_items[1]
            for row, row_arrows in enumerate(y_arrows):
                for col, arrow in enumerate(row_arrows):
                    # Bar middle position
                    x, y = (int((self.vrtx_coords[row][col][0] + self.vrtx_coords[row + 1][col][0])/2),
                            int((self.vrtx_coords[row][col][1] + self.vrtx_coords[row + 1][col][1])/2))
                    
                    # If mask is dark in this area, save state for statistics
                    if self.transformed_mask[x, y] < 127: # Flipped x and y here
                        written_y_arrows.append(arrow.getText())


            ## CALCULATE STATISTICS
            shifted_options = self.vrtx_options[1:] + [self.vrtx_options[0]] # Shifting undefined to the end for readability
            vrtx_counts = []
            for state in shifted_options:
                vrtx_counts.append(written_vertices.count(state))
            vrtx_freq = np.array(vrtx_counts)/np.sum(vrtx_counts)
            
            x_counts = []
            for state in self.h_options:
                x_counts.append(written_x_arrows.count(state))
            x_og_num = x_counts.pop(self.params["Default x-bar saturation: ← = 1 or → = 2"].value)
            x_flipped = x_counts[1]/(x_og_num + x_counts[1])
            
            y_counts = []
            for state in self.v_options:
                y_counts.append(written_y_arrows.count(state))
            y_og_num = y_counts.pop(self.params["Default y-bar saturation: ↑ = 1 or ↓ = 2"].value)
            y_flipped = y_counts[1]/(y_og_num + y_counts[1])

            # vertex options: ['T1', 'T2_og', 'T2', 'T3.1', 'T3.2', 'T4.1', 'T4.1', 'U']
            vrtx_freq_grouped = [vrtx_freq[0],
                                 vrtx_freq[1] + vrtx_freq[2],
                                 vrtx_freq[3] + vrtx_freq[4],
                                 vrtx_freq[5] + vrtx_freq[6],
                                 vrtx_freq[7]]
            vrtx_out = [0, 0, vrtx_freq[3], vrtx_freq[5], 0]
            vrtx_unwritten = [0, vrtx_freq[1], 0, 0, 0]

            state_labels = []
            for elem in shifted_options:
                if elem[:2] not in state_labels:
                    state_labels.append(elem[:2])

            # Frequencies for thermal model
            thermal_model = [2, 4, 8, 2, 0]
            thermal_freq = np.array(thermal_model)/np.sum(thermal_model)

            x = range(len(state_labels))
            width = 0.35

            fig, ax = plt.subplots()

            filename = self.params["Filename"].value
            for i, char in enumerate(filename):
                if char == "/":
                    cut = i

            ax.bar(x, vrtx_freq_grouped, width, label = filename[cut + 1:cut + 21], color = 'blue')
            ax.bar(x, vrtx_out, width, label = 'Out states', color = 'darkblue')
            ax.bar(x, vrtx_unwritten, width, label = 'Unwritten states', color = 'g')
            ax.bar([p + width for p in x], thermal_freq, width, label = 'Thermal model', color = 'red')

            ax.set_xlabel('Vortex Type')
            ax.set_ylabel('Frequency')
            ax.set_xticks([p + width/2 for p in x])
            ax.set_xticklabels(state_labels)
            ax.legend()
            ax.set_title(f"x-bars: {x_counts[1]} flipped, {x_og_num} not flipped ({x_flipped*100:.2f}%)\ny-bars: {y_counts[1]} flipped, {y_og_num} not flipped ({y_flipped*100:.2f}%)")

            plt.savefig(filename + '_statistics.png')
            self.params['Latest error'].set_value('Statistics plot saved')

        else:
            self.params['Latest error'].set_value('Evaluate vertices and load mask first')


def main():
    reload(fv)
    # shell = get_ipython()
    app = QApplication(sys.argv)
    w = fv.ManyFilesViewer(ScanView)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
