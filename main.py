import sys, os
import pickle

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenu, QToolBar, QToolButton, \
    QAction, QActionGroup, QTextEdit, QWidget, QListWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QMessageBox, QScrollArea, \
    QSplashScreen, QProgressBar, QSpacerItem, QHBoxLayout, QTextEdit, QFileDialog, QListWidgetItem, QAbstractItemView, \
    QStatusBar, QGridLayout, QDial, QToolBox, QRadioButton, QGroupBox, QSpinBox,QDoubleSpinBox, QTabWidget, QComboBox, QPlainTextEdit, \
    QInputDialog

from PyQt5.QtGui import QCursor, QPixmap, QIcon

from PyQt5 import QtGui
from PyQt5 import uic, QtCore

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import xmltodict
import copy

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

try:
    import gdal, ogr
except ModuleNotFoundError:
    from osgeo import gdal, ogr

from scipy.interpolate import LinearNDInterpolator    

# set proper current directory
current_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(current_dir)

# import user function
from pyLEFA_Functions import gray2binary, detectPLineHough, detectPLineHough2, uniteLines2, lineKB, \
    createSHPfromDictionary, adjusted_canny, \
    saveLinesShpFile, saveLinesShpFile2, rasterize_shp, rasterize_shp3, saveGeoTiff, get_minkowski, \
    get_pixels_sum, get_average_val, build_rose_diag, get_probability_matrix, get_shp_extent

from pyLEFA_Functions import ProgressBar, GetExtent

# TODO Data structure, convert data structure into XML file
"""
Структура данных массива открытых растров как словарь списков:
Data structure of opened raster arrays as dictionary list
{'id':____} unique integer number autoincrement 
{'name':____} filename without extension
{'file_path':____} file path full
{'type':____} tif shp 
{'parent':____} none if opened from disk or created from no sources, otherwise contains integer ID of parent file
{'gdal_obj':____} gdal object for georef data, none if nogeoref 
{'draw_order':____} integer for obj drawing order
{'content': ____} description of generated file (lines, faults, densmap)
"""


# TODO data unit object Dataobj
class Dataobj():
    def __init__(self, id=-1, name='noname', file_path='', ftype=None, parent_obj=None, \
                 gt=None, gdal_obj=None, data=None, draw_order=-1, content=None):
        self.id = id  # unique id
        self.file_path = file_path  # file path full
        self.type = ftype  # tif or shp
        self.parent_obj = parent_obj  # none if opened from disk or created from no sources, otherwise contains integer ID
        print('new Dataobj')
        print('parent_obj=', parent_obj)
        print('self.parent_obj=', self.parent_obj)
        self.gdal_obj = gdal_obj  # gdal object for georef data, none if nogeoref (defaults)
        self.draw_order = draw_order  # integer for obj drawing order
        self.band = None  # by default there is no band
        self.content = content  # content of data object lines, faults
        if data == None and gdal_obj != None:  # works for tiffs
            band = gdal_obj.GetRasterBand(1)
            self.data = band.ReadAsArray()
        else:
            self.data = data

        file_path_name = os.path.split(file_path)
        if 'tif' in file_path_name[-1].split('.')[-1] or self.type == 'tif':
            self.type = 'tif'
        elif 'shp' in file_path_name[-1].split('.')[-1] or self.type == 'shp':
            self.type = 'shp'
        else:
            print('Unknown datafile type. Set it to tif by default.')
            self.type = 'tif'  # set it to tif anyway

        self.name = file_path_name[-1].split('.')[-2]  # filename without  extension

        if self.type == 'shp':
            self.gt = gt  # TODO Geo transform (for Shapefiles only)
            # TODO Read feature data from SHP
            self.data = self.get_shp_features(self.file_path)  # {'extent':ext, 'features':features}

    def get_band_from_gdal(self, gdal_object):
        band = gdal_object.GetRasterBand(1)
        rasterData = band.ReadAsArray()
        return rasterData

    def get_geo_transform(self):
        if self.gdal_obj != None:
            gt = self.gdal_obj.GetGeoTransform()
            return gt
        else:
            return None

    # TODO shape file feature extraction
    def get_shp_features(self, file_path_name):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(file_path_name, 0)  # 0 means read-only. 1 means writeable.
        layer = dataSource.GetLayer()
        ext = layer.GetExtent()
        featureCount = layer.GetFeatureCount()
        print('total features', featureCount)
        type_str = None
        features = []
        for feat in layer:
            geom = feat.GetGeometryRef()
            if 'line' in geom.ExportToWkt().lower():
                type_str = 'line'
                # print(geom)
                points = geom.GetPointCount()
                vertices = []
                for p in range(points):
                    lon, lat, z = geom.GetPoint(p)
                    # print([lon, lat])
                    vertices.append([lon, lat])
                features.append(vertices)
            elif 'poly' in geom.ExportToWkt().lower():
                type_str = 'poly'
                ring = geom.GetGeometryRef(0)
                points = ring.GetPointCount()
                vertices = []
                for p in range(points):
                    lon, lat, z = ring.GetPoint(p)
                    # print([lon, lat])
                    vertices.append([lon, lat])
                features.append(vertices)
            elif 'point' in geom.ExportToWkt().lower():
                type_str = 'point'
                # print('point features arent supported')
                # self.msg_info('Point features arent supported')
                vertices = []
                lon, lat = geom.GetX(), geom.GetY()
                vertices.append([lon, lat])
                features.append(vertices)

        return {'extent': ext, 'features': features, 'type': type_str}


class DataobjRaster():
    def __init__(self, id=-1, name='noname', file_path='', ftype=None, parent=None, gdal_obj=None, draw_order=-1):
        self.id = id  # unique id
        self.name = name  # filename without  extension
        self.file_path = file_path  # file path full
        self.type = ftype  # tif or shp
        self.parent = parent  # none if opened from disk or created from no sources, otherwise contains integer ID
        self.gdal_obj = gdal_obj  # gdal object for georef data, none if nogeoref (defaults)
        self.draw_order = draw_order  # integer for obj drawing order
        self.band = []
        self.rc = []


# TODO DataobjVector
class DataobjVector():
    def __init__(self, id=-1, name='noname', file_path='', ftype=None, parent=None, gdal_obj=None, draw_order=-1):
        self.id = id  # unique id
        self.name = name  # filename without  extension
        self.file_path = file_path  # file path full
        self.type = ftype  # tif or shp
        self.parent = parent  # none if opened from disk or created from no sources, otherwise contains integer ID
        self.gdal_obj = gdal_obj  # gdal object for georef data, none if nogeoref (defaults)
        self.draw_order = draw_order  # integer for obj drawing order
        self.band = []
        self.rc = []


# TODO data storage object
class DataStorage():
    def __init__(self, parent=None):
        self.list_objects = []  # list of objects in the storage
        self.parent = parent

    def get_new_id(self):
        id_list = []
        for obj in self.list_objects:
            id_list.append(obj.__dict__['id'])
        if len(id_list) == 0:
            return 0
        else:
            return max(id_list) + 1

    def delete_obj(self, id):
        for obj in self.list_objects:
            if obj['id'] == id:
                self.list_objects.remove(obj)

    def delete_obj_byname(self, name):
        for obj in self.list_objects:
            if obj.name == name:
                self.list_objects.remove(obj)

    def add_obj(self, obj: Dataobj):
        # check if such filepath exists in the storage
        is_unique = 1
        for eobj in self.list_objects:
            if eobj.file_path == obj.file_path:
                is_unique = 0
        if is_unique == 1:
            self.list_objects.append(obj)
        else:
            print('objest exists in storage')
            self.parent.msg_info(msg='objest exists in storage')

    def get_objects(self):
        return self.list_objects

    def open_data(self, filepath, parent_obj=None, content=None):
        # TODO add additional parameters in project loading clause
        if 'tif' in os.path.split(filepath)[-1]:
            try:
                gdal_obj = gdal.Open(filepath)
                # create new object and put it to the storage
                dobj = Dataobj(id=self.get_new_id(), gdal_obj=gdal_obj, file_path=filepath, ftype='tif',
                               parent_obj=parent_obj, content=content)
                self.add_obj(dobj)
            except:
                print('Can not open data file', filepath, '. Unknown error has occurred.')

        elif 'shp' in os.path.split(filepath)[-1]:
            # print('Shape file opening is not supported yet')
            # TODO Shape file opening is not supported yet
            dobj = Dataobj(id=self.get_new_id(), gdal_obj=None, file_path=filepath, ftype='shp', parent_obj=parent_obj,
                           content=content)
            self.add_obj(dobj)


class Window(QMainWindow):
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)

        self.resize(700, 70)

        self.settings = QtCore.QSettings('FEGI', 'pyLEFA0.62a')

        # default settings dictionary
        defaults = {
            'binarization': 'none',
            'line_sigma': 1,
            'line_tresh': 5,
            'line_length': 3,
            'line_gap': 0,
            'fault_tresh': 30,
            'fault_length': 50,
            'fault_gap': 3,
            'density_window_size': 100,
            'fractal_window_size': 100,
            'raster_res': 10,
            'raster_radius': 1000,
            'canny_sigma':10,
            'canny_lowt':10,
            'canny_hight':30
        }

        if not self.settings.contains("data"):
            print('reload settings from defaults')
            self.settings.setValue('data', defaults)

        # reload defaults if new fields were added
        for key in defaults:
            if key not in self.settings.value('data'):
                # reset
                print('reset settings to defaults')
                self.settings.setValue('data', defaults)

        # TODO output rasters only (no contours) just for compatibility sake
        self.output = 'raster'

        # TODO project file path
        self.preset_folder = os.path.join('presets')
        self.preset_lines_folder = os.path.join('presets', 'lines')
        self.preset_faults_folder = os.path.join('presets', 'faults')
        # TODO check if preset folder doesn't exist, create it
        if not os.path.exists(self.preset_folder):
            os.mkdir(self.preset_folder)
        if not os.path.exists(self.preset_lines_folder):
            os.mkdir(self.preset_lines_folder)
        if not os.path.exists(self.preset_faults_folder):
            os.mkdir(self.preset_faults_folder)


        # name of language settings file
        self.localization = 'localization.xml'
        self.settings_lang_file = 'language.sav'  # store saved
        self.selected_language = 'eng'  # default language

        #
        # try to load and parse language file
        self.language_dict = []
        try:
            with open(self.localization, 'r', encoding='utf-8') as file:
                my_xml = file.read()
            my_dict = xmltodict.parse(my_xml)
            self.language_dict = my_dict['body']
        except:
            QMessageBox.critical(self, 'Error loading language file.',
                                 'Language file could not be loaded!',
                                 QMessageBox.Ok, QMessageBox.Ok)
            self.close()
        self.lang_actions = {}
        # try to load save language file

        self._createActions()
        self._createMenuBar()
        self._createToolBars()
        self._createContextMenu()

        # bind menu actions and functions
        self.exitAction.triggered.connect(self.close)

        
        # layers icons
        self.pict_vect_layer = os.path.join('resources', 'icovlayer.png')
        self.pict_pnt_layer = os.path.join('resources', 'icoplayer.png')
        self.pict_rast_layer = os.path.join('resources', 'icorlayer.png')
        self.pict_unk_layer = os.path.join('resources', 'icoulayer.png')

        # TODO !!!flags and parameters for analysis underway
        self.project_file_path = None
        self.current_method = None  # line, faults, density, minkowski
        self.file_for_analysis = None  # input file for performing analysis on (from layer list)
        self.file_for_analysis_target = None
        self.save_tif_file_path = None  # out
        self.save_shp_file_path = None  # out
        self.pref_binary = 'Canny'  # flow (thalwegs)
        self.pref_threshold = 100
        self.pref_line_length = 10
        self.pref_lgap = 2
        self.infile = ''
        self.outfile = ''
        self.stat_window = []
        # app variables for DATA storing
        # TODO data structure declaration
        self.data_structure = []

        self.data_storage = DataStorage(parent=self)  # TODO try to use data storage for every layer

        self.gdal_object = []  # object for storing of gdal
        self.rows = []
        self.cols = []
        self.srtm = []  # raster relief
        self.grid = []  # grid
        self.inflated_dem = []  # dem without sinkholes and flats
        self.gdal_object = []  # gdal object
        self.flow_directions = []  # accumulated flows
        self.flow_orders = []  # flow orders raster dataset
        self.base_surfaces_dict = {}  # dictionary for storing base surfaces
        self.base_surfaces_diff_dict = {}  # dictionary for storing base surface differences
        self.xgrid = []  #
        self.ygrid = []
        self.report_dir = []
        self.report_dir_image_subdir = 'img'
        self.report_file_ext = '.png'
        self.report_file_name = 'report.html'
        self.report_window = []

        # create application splashscreen
        self.splash = QSplashScreen(QPixmap("splash.png"))

        # progress bar in status bar
        self.pbar = QProgressBar()
        self.si = QSpacerItem(100, 25)  # buffer objects
        self.pbar.setGeometry(0, 0, 600, 25)

        # statusbar строка состояния
        self.statusbar = QStatusBar()
        self._statusbar_label = QLabel('Processing...')
        self.statusbar.addPermanentWidget(self._statusbar_label)
        self.statusbar.addPermanentWidget(self.pbar)
        # hide pbar widgets
        self.pbar.hide()
        self._statusbar_label.hide()

        

        # add language group into settings menu
        language_group = QActionGroup(self)

        # check if language settings file exists
        if not os.path.isfile(self.settings_lang_file):
            print('language file doesnt exist! Saving defaults...')
            with open(self.settings_lang_file, 'w') as f:
                f.write(self.selected_language)
        else:
            print('language file exists! loading...')
            with open(self.settings_lang_file, 'r') as f:
                self.selected_language = f.read()
            # reload language
            self.reset_language()

        # set window title
        title = self.language_dict['commands']['maint_title'][self.selected_language]
        self.setWindowTitle(title)

        # populate language menu items
        for key in [*self.language_dict['languages']]:
            print(self.language_dict['languages'][key])

            # TODO MENU options
            menu_element = language_group.addAction(self.language_dict['languages'][key])
            self.menuLanguage.addAction(menu_element)
            menu_element.setCheckable(True)

            if key == self.selected_language:  # if this language is selected
                menu_element.setChecked(True)
            # menu_element.triggered.connect(lambda:self.choose_language(key)) #set action for menu element
            self.lang_actions.update({key: menu_element})  # add lang action to menu

        for key in [*self.lang_actions]:
            self.lang_actions[key].triggered.connect(lambda checked, arg=key: self.choose_language(arg))

        # TODO  create subwindows in class Window
        self.layers = Layers(parent=self)  # layers window
        self.map_browser = MapBrowser(parent=self)  # browser window
        # self.select_file_analysis = selectFileWindow(parent=self)
        self.preferences = Preferences(parent=self)  # linemaents pref window

    def showStatWin(self, msg='', title=''):
        self.stat_window = StatWindow(txt=msg, title=title)
        self.stat_window.show()

    def saveSettings(self, set_dict=None, key=None, val=None):
        saved_dict = self.settings.value('data')

        if dict == None and key == None and val == None:
            print('Nothing to save')
        elif key != None and val != None:
            print('Save key/value pair')
            if key in [*saved_dict]:
                saved_dict[key] = val
            else:
                saved_dict.update({key: val})
            self.settings.setValue('data', saved_dict)

        elif set_dict != None:
            print('Save dictionary setting')
            for key in [*set_dict]:
                if key in [*saved_dict]:
                    saved_dict[key] = set_dict[key]
                else:
                    saved_dict.update({'key': set_dict[key]})
            self.settings.setValue('data', saved_dict)
        else:
            print('Unknown choice')

    # project save
    def dump_project(self, path):
        # with open(path, 'wb') as handle:
        #     pickle.dump(self.data_storage, handle)
        self.datastorage2xml(path)

    # project load
    def load_project(self, path):
        somedict = self.xml2project(path)
        for key in [*somedict['body']]:
            self.add_data_obj(somedict['body'][key]['file_path'], parent_obj=somedict['body'][key]['parent'])
        self.update_list2storage()

    def generate_heatmap(self, point_file_name, raster_pixel_size, raster_radius, aoi_extent=None):
        # extent of selected point shp file
        points = self.get_features_by_name_id(name=point_file_name)
        extent = self.get_extent_by_name_id(name=point_file_name)
        shp_file_path = self.get_path_by_name_id(name=point_file_name)
        xlist = [pnt[0][0] for pnt in points]
        ylist = [pnt[0][1] for pnt in points]
        if aoi_extent is not None:
            x = [xval - aoi_extent[0] for xval in xlist]
            y = [yval - aoi_extent[2] for yval in ylist]
            xminmax, yminmax = [min(xlist)-aoi_extent[0], aoi_extent[1]-aoi_extent[0]], [min(ylist)-aoi_extent[2], aoi_extent[3]-aoi_extent[2]]
            #width = int((aoi_extent[1] - aoi_extent[0]) / raster_pixel_size[0])
            #heigth = int((aoi_extent[3] - aoi_extent[2]) / raster_pixel_size[1])
            #print('extent=',aoi_extent)
            # print('W0 pix=', width)
            # print('H0 pix=', heigth)
            # print('W pix=',(xminmax[1]-xminmax[0])//raster_pixel_size[0])
            # print('H pix=',(yminmax[1]-yminmax[0])//raster_pixel_size[1])

        else:
            x = [xval - min(xlist) for xval in xlist]
            y = [yval - min(ylist) for yval in ylist]
            xminmax, yminmax = [min(x), max(x)], [min(y), max(y)]

        # DEFINE GRID SIZE AND RADIUS(h)
        heatmap = get_probability_matrix(x, y, xminmax, yminmax, grid_size=raster_pixel_size, h=raster_radius,aoi_extent=aoi_extent)
        return np.flipud(heatmap)

    # select project file
    def new_project_file(self):
        txt = self.language_dict['commands']['msg_new_proj'][self.selected_language]
        if self.msg_yn(msg=txt) == 'y':
            self.data_storage = DataStorage(parent=self)
            self.update_list2storage()
            self.map_browser.close()
            print('project aws resetted')

    def save_project_file(self):
        if self.project_file_path == None:
            path = QFileDialog.getSaveFileName(self, ("Save project"), '', ("lefa proj (*.lefa)"))
            if path[0] != '':
                self.project_file_path = path[0]
        self.dump_project(self.project_file_path)
        print('Project has been saved to file', self.project_file_path)

    def save_project_file_as(self):
        path = QFileDialog.getSaveFileName(self, ("Save project as"), '', ("lefa proj (*.lefa)"))
        if path[0] != '':
            self.project_file_path = path[0]
            self.dump_project(self.project_file_path)

    def open_project_file(self):
        path = QFileDialog.getOpenFileName(self, ("Open project file"), '', ("lefa proj (*.lefa)"))
        if path[0] != '':
            self.project_file_path = path[0]
            self.load_project(self.project_file_path)

    def save_tif_file(self, txt="Save raster as"):
        path = QFileDialog.getSaveFileName(self, (txt), '', ("geotiff (*.tif)"))
        if path[0] != '':
            self.save_tif_file_path = path[0]
            return path[0]
        else:
            return None

    def save_shp_file(self, txt="Save vector as"):
        path = QFileDialog.getSaveFileName(self, (txt), '', ("shape (*.shp)"))
        if path[0] != '':
            self.save_shp_file_path = path[0]
            return path[0]
        else:
            return None

    def _createMenuBar(self):
        # menu bar on top of the window
        self.menuBar = self.menuBar()
        self.fileMenu = QMenu('&File', self)
        self.menuBar.addMenu(self.fileMenu)
        self.fileMenu.addAction(self.newAction)
        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.saveAsAction)
        self.fileMenu.addAction(self.exitAction)
        # editMenu = menuBar.addMenu("&Edit")  #TODO replace to layer manipulations
        # editMenu.addAction(self.copyAction)
        # editMenu.addAction(self.pasteAction)
        # editMenu.addAction(self.cutAction)
        # findMenu = editMenu.addMenu("Find and Replace")
        # findMenu.addAction("Find...")
        # findMenu.addAction("Replace...")
        self.viewMenu = self.menuBar.addMenu("&View")
        self.viewMenu.addAction(self.viewProjectFiles)
        self.viewMenu.addAction(self.viewPreferences)
        # editMenu.addAction(self.pasteAction)
        # editMenu.addAction(self.cutAction)
        # findMenu = editMenu.addMenu("Find and Replace")
        # findMenu.addAction("Find...")
        # findMenu.addAction("Replace...")
        self.menuLanguage = self.menuBar.addMenu('&Language')
        self.helpMenu = self.menuBar.addMenu('&Help')
        # helpMenu.addAction(self.helpContentAction)
        self.helpMenu.addAction(self.aboutAction)
        self.menuBar.setNativeMenuBar(False)  # отключаем вывод меню как в операционной системе

    def choose_language(self, lang):
        # saving default language into self.settings_lang_file
        self.selected_language = lang
        print('select language', lang)
        with open(self.settings_lang_file, 'w') as f:  # save to disk
            f.write(self.selected_language)
        self.msg_info(self.language_dict['commands']['msg_reset_lang'][self.selected_language])
        self.reset_language()

    def reset_language(self):
        # reset all languages in main window
        self.fileMenu.setTitle(self.language_dict['commands']['menu_file'][self.selected_language])
        self.viewMenu.setTitle(self.language_dict['commands']['menu_view_view'][self.selected_language])
        self.viewMenu.setTitle(self.language_dict['commands']['menu_view_view'][self.selected_language])
        self.menuLanguage.setTitle(self.language_dict['commands']['menu_language'][self.selected_language])
        self.helpMenu.setTitle(self.language_dict['commands']['menu_help'][self.selected_language])

        # actions
        self.newAction.setText(self.language_dict['commands']['menu_file_new'][self.selected_language])
        self.openAction.setText(self.language_dict['commands']['menu_fileopen'][self.selected_language])
        self.saveAction.setText(self.language_dict['commands']['menu_file_save'][self.selected_language])
        self.saveAsAction.setText(self.language_dict['commands']['menu_file_saveas'][self.selected_language])
        self.exitAction.setText(self.language_dict['commands']['menu_fileexit'][self.selected_language])
        self.viewProjectFiles.setText(self.language_dict['commands']['menu_view_files'][self.selected_language])
        self.viewPreferences.setText(self.language_dict['commands']['menu_view_settings'][self.selected_language])
        self.aboutAction.setText(self.language_dict['commands']['menu_about_lefa'][self.selected_language])

    def _createContextMenu(self):
        pass;
        # self.centralWidget.setContextMenuPolicy(Qt.ActionsContextMenu)
        # self.centralWidget.addAction(self.newAction)
        # self.centralWidget.addAction(self.openAction)
        # self.centralWidget.addAction(self.saveAction)
        # self.centralWidget.addAction(self.copyAction)
        # self.centralWidget.addAction(self.pasteAction)
        # self.centralWidget.addAction(self.cutAction)

    def _createToolBars(self):
        # toolbars
        fileToolBar = self.addToolBar("File")
        editToolBar = QToolBar("Edit", self)

        # buttons
        new_project_button = QToolButton(self)
        new_project_button_icon = QtGui.QIcon()
        new_project_button_icon_path = os.path.join('resources', 'b_new.png')
        new_project_button_icon.addPixmap(QtGui.QPixmap(new_project_button_icon_path), QtGui.QIcon.Normal,
                                          QtGui.QIcon.Off)
        new_project_button_tooltip = self.language_dict['commands']['new_project_button_tooltip'][self.selected_language]

        save_project_button = QToolButton(self)
        save_project_button_icon = QtGui.QIcon()
        save_project_button_icon_path = os.path.join('resources', 'b_save.png')
        save_project_button_icon.addPixmap(QtGui.QPixmap(save_project_button_icon_path), QtGui.QIcon.Normal,
                                           QtGui.QIcon.Off)
        save_project_button_tooltip = self.language_dict['commands']['save_project_button_tooltip'][self.selected_language]

        open_project_button = QToolButton(self)
        open_project_button_icon = QtGui.QIcon()
        open_project_button_icon_path = os.path.join('resources', 'b_open.png')
        open_project_button_icon.addPixmap(QtGui.QPixmap(open_project_button_icon_path), QtGui.QIcon.Normal,
                                           QtGui.QIcon.Off)
        open_project_button_tooltip = self.language_dict['commands']['open_project_button_tooltip'][self.selected_language]

        open_datasource_button = QToolButton(self)
        open_datasource_button_icon = QtGui.QIcon()
        open_datasource_button_path = os.path.join('resources', 'b_opendata.png')
        open_datasource_button_icon.addPixmap(QtGui.QPixmap(open_datasource_button_path), QtGui.QIcon.Normal,
                                              QtGui.QIcon.Off)
        open_datasource_button_tooltip = self.language_dict['commands']['open_rastervector_button_tooltip'][self.selected_language]

        open_shp_button = QToolButton(self)
        open_shp_button_icon = QtGui.QIcon()
        open_shp_button_icon_path = os.path.join('resources', 'b_openvect.png')
        open_shp_button_icon.addPixmap(QtGui.QPixmap(open_shp_button_icon_path), QtGui.QIcon.Normal,
                                       QtGui.QIcon.Off)
        open_shp_button_tooltip = self.language_dict['commands']['open_shp_button_tooltip'][self.selected_language]

        #detect_edges_canny_button
        detect_edges_canny_button = QToolButton(self)
        detect_edges_canny_button_icon = QtGui.QIcon()
        detect_edges_canny_button_icon_path = os.path.join('resources', 'b_canny.png')
        detect_edges_canny_button_icon.addPixmap(QtGui.QPixmap(detect_edges_canny_button_icon_path), QtGui.QIcon.Normal,
                                         QtGui.QIcon.Off)
        detect_line_button_tooltip = self.language_dict['commands']['detect_canny_button_tooltip'][self.selected_language]

        rasterdensmap_canny_button = QToolButton(self)
        rasterdensmap_canny_button_icon = QtGui.QIcon()
        rasterdensmap_canny_button_icon_path = os.path.join('resources', 'b_raster_densmap.png')
        rasterdensmap_canny_button_icon.addPixmap(QtGui.QPixmap(rasterdensmap_canny_button_icon_path), QtGui.QIcon.Normal,
                                         QtGui.QIcon.Off)
        rasterdensmap_canny_button_tooltip = self.language_dict['commands']['rasterdensmap_canny_button_tooltip'][self.selected_language]

        detect_line_button = QToolButton(self)
        detect_line_button_icon = QtGui.QIcon()
        detect_line_button_icon_path = os.path.join('resources', 'b_detectlines.png')
        detect_line_button_icon.addPixmap(QtGui.QPixmap(detect_line_button_icon_path), QtGui.QIcon.Normal,
                                          QtGui.QIcon.Off)
        detect_line_button_tooltip = self.language_dict['commands']['detect_line_button_tooltip'][self.selected_language]

        detect_fault_button = QToolButton(self)
        detect_fault_button_icon = QtGui.QIcon()
        detect_fault_button_icon_path = os.path.join('resources', 'b_detectfaults.png')
        detect_fault_button_icon.addPixmap(QtGui.QPixmap(detect_fault_button_icon_path), QtGui.QIcon.Normal,
                                           QtGui.QIcon.Off)
        detect_fault_button_tooltip = self.language_dict['commands']['detect_fault_button_tooltip'][self.selected_language]

        compute_line_dens_button = QToolButton()
        compute_line_dens_button_icon = QtGui.QIcon()
        compute_line_dens_button_icon_path = os.path.join('resources', 'b_density.png')
        compute_line_dens_button_icon.addPixmap(QtGui.QPixmap(compute_line_dens_button_icon_path), QtGui.QIcon.Normal,
                                                QtGui.QIcon.Off)
        compute_line_dens_button_tooltip = self.language_dict['commands']['compute_line_dens_button_tooltip'][self.selected_language] 

        compute_minkowski_button = QToolButton()
        compute_minkowski_button_icon = QtGui.QIcon()
        compute_minkowski_button_icon_path = os.path.join('resources', 'b_calc_minkowski.png')
        compute_minkowski_button_icon.addPixmap(QtGui.QPixmap(compute_minkowski_button_icon_path), QtGui.QIcon.Normal,
                                                QtGui.QIcon.Off)
        compute_minkowski_button_tooltip = self.language_dict['commands']['compute_minkowski_button_tooltip'][self.selected_language] 

        compute_rose_button = QToolButton()
        compute_rose_button_icon = QtGui.QIcon()
        compute_rose_button_icon_path = os.path.join('resources', 'b_rose.png')
        compute_rose_button_icon.addPixmap(QtGui.QPixmap(compute_rose_button_icon_path), QtGui.QIcon.Normal,
                                           QtGui.QIcon.Off)
        compute_rose_button_tooltip = self.language_dict['commands']['compute_rose_button_tooltip'][self.selected_language] 

        compute_heatmap_button = QToolButton()
        compute_heatmap_button_icon = QtGui.QIcon()
        compute_heatmap_button_icon_path = os.path.join('resources', 'b_heatmap.png')
        compute_heatmap_button_icon.addPixmap(QtGui.QPixmap(compute_heatmap_button_icon_path), QtGui.QIcon.Normal,
                                              QtGui.QIcon.Off)
        compute_heatmap_button_tooltip = self.language_dict['commands']['compute_heatmap_button_tooltip'][self.selected_language] 

        compute_datatable_button = QToolButton()
        compute_datatable_button_icon = QtGui.QIcon()
        compute_datatable_button_icon_path = os.path.join('resources', 'b_data.png')
        compute_datatable_button_icon.addPixmap(QtGui.QPixmap(compute_datatable_button_icon_path), QtGui.QIcon.Normal,
                                                QtGui.QIcon.Off)
        compute_datatable_button_tooltip =self.language_dict['commands']['compute_datatable_button_tooltip'][self.selected_language] 

        # adding icon to the toolbuttonss
        new_project_button.setIcon(new_project_button_icon)
        new_project_button.clicked.connect(self.new_project_file)
        new_project_button.setToolTip(new_project_button_tooltip)

        save_project_button.setIcon(save_project_button_icon)
        save_project_button.clicked.connect(self.save_project_file)
        save_project_button.setToolTip(save_project_button_tooltip)

        open_project_button.setIcon(open_project_button_icon)
        open_project_button.clicked.connect(self.open_project_file)
        open_project_button.setToolTip(open_project_button_tooltip)

        open_datasource_button.setIcon(open_datasource_button_icon)
        open_datasource_button.clicked.connect(lambda: self.file_open_dialogue(type_filter='tif'))
        open_datasource_button.setToolTip(open_datasource_button_tooltip)

        # create_shp_button.setIcon(create_shp_button_icon)
        # create_shp_button.clicked.connect(self.testFunction)
        # create_shp_button.setToolTip(create_shp_button_tooltip)

        open_shp_button.setIcon(open_shp_button_icon)
        open_shp_button.clicked.connect(lambda: self.file_open_dialogue(type_filter='shp'))
        open_shp_button.setToolTip(open_shp_button_tooltip)

        detect_edges_canny_button.setIcon(detect_edges_canny_button_icon)
        detect_edges_canny_button.clicked.connect(self.analysis_detect_edges)
        detect_edges_canny_button.setToolTip(detect_line_button_tooltip)

        rasterdensmap_canny_button.setIcon(rasterdensmap_canny_button_icon)
        rasterdensmap_canny_button.clicked.connect(self.create_rasterdensmap)
        rasterdensmap_canny_button.setToolTip(rasterdensmap_canny_button_tooltip)

        detect_line_button.setIcon(detect_line_button_icon)
        detect_line_button.clicked.connect(self.analysis_detect_lines)
        detect_line_button.setToolTip(detect_line_button_tooltip)

        detect_fault_button.setIcon(detect_fault_button_icon)
        detect_fault_button.clicked.connect(self.analysis_detect_faults)
        detect_fault_button.setToolTip(detect_fault_button_tooltip)

        # TODO M and D
        compute_line_dens_button.setIcon(compute_line_dens_button_icon)
        compute_line_dens_button.clicked.connect(self.analysis_density)
        compute_line_dens_button.setToolTip(compute_line_dens_button_tooltip)

        compute_minkowski_button.setIcon(compute_minkowski_button_icon)
        compute_minkowski_button.clicked.connect(self.analysis_minkowski)
        compute_minkowski_button.setToolTip(compute_minkowski_button_tooltip)

        # TODO add procedure to rose button
        compute_rose_button.setIcon(compute_rose_button_icon)
        compute_rose_button.clicked.connect(self.analysis_rose_diagram)
        compute_rose_button.setToolTip(compute_rose_button_tooltip)

        compute_heatmap_button.setIcon(compute_heatmap_button_icon)
        compute_heatmap_button.clicked.connect(self.analysis_heatmap)
        compute_heatmap_button.setToolTip(compute_heatmap_button_tooltip)

        compute_datatable_button.setIcon(compute_datatable_button_icon)
        compute_datatable_button.clicked.connect(self.analysis_datatable)
        compute_datatable_button.setToolTip(compute_datatable_button_tooltip)

        # add buttons to toolbars
        fileToolBar.addWidget(new_project_button)
        fileToolBar.addWidget(save_project_button)
        fileToolBar.addWidget(open_project_button)
        fileToolBar.addWidget(open_datasource_button)
        fileToolBar.addWidget(open_shp_button)

        # fileToolBar.addWidget(open_datasource_button)
        editToolBar.addWidget(detect_edges_canny_button)
        editToolBar.addWidget(detect_line_button)
        editToolBar.addWidget(detect_fault_button)
        editToolBar.addWidget(compute_minkowski_button)
        editToolBar.addWidget(compute_line_dens_button)
        editToolBar.addWidget(compute_rose_button)
        editToolBar.addWidget(compute_heatmap_button)
        editToolBar.addWidget(rasterdensmap_canny_button)
        editToolBar.addWidget(compute_datatable_button)
        self.addToolBar(editToolBar)

        # add toolbars to main window
        # context menu

        # helpToolBar = QToolBar("Help", self)
        # self.addToolBar(Qt.LeftToolBarArea, helpToolBar)
        # self.addToolBar(Qt.BottomToolBarArea, helpToolBar)

    def _createActions(self):
        self.newAction = QAction(self)
        self.newAction.setText('New project')
        self.openAction = QAction('Open project', self)
        self.saveAction = QAction('Save project', self)
        self.saveAsAction = QAction('Save project as', self)
        self.exitAction = QAction('Exit', self)
        self.copyAction = QAction("&Copy", self)
        self.pasteAction = QAction("Paste", self)
        self.cutAction = QAction("Cut", self)
        self.viewProjectFiles = QAction('View files')
        self.viewPreferences = QAction('Settings', self)
        # self.helpContentAction = QAction("Help", self)
        self.aboutAction = QAction('About', self)

        # connect actions and processing functions
        self.newAction.triggered.connect(self.new_project_file)
        self.openAction.triggered.connect(self.open_project_file)
        self.saveAction.triggered.connect(self.save_project_file)
        self.saveAsAction.triggered.connect(self.save_project_file_as)
        self.viewProjectFiles.triggered.connect(self.showLayersPane)
        self.viewPreferences.triggered.connect(self.showPreferences)
        self.aboutAction.triggered.connect(self.about_dialogue)

    def showPreferences(self):
        print('Preferences window was called')
        self.preferences.show()

    def showLayersPane(self):
        self.layers.show()

    def testFunction(self):
        print('you have toggled test function, bruh')
        self.layers.show()

    def msg_yn(self, msg='Are you sure?'):
        msg_title = self.language_dict['commands']['msg_yn_title'][self.selected_language]
        msgBox = QMessageBox.question(self, msg_title, msg,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes: return 'y'
        if msgBox == QMessageBox.No: return 'n'

    def msg_info(self, msg='some FYI'):
        msg_title = 'Warning message:'
        msgBox = QMessageBox.information(self, msg_title, msg,
                                         QMessageBox.Ok, QMessageBox.Ok)

    def closeEvent(self, event):
        msg_title = self.language_dict['commands']['msg_specify_close_title'][self.selected_language]
        msg = self.language_dict['commands']['msg_specify_close_text'][self.selected_language]
        msgBox = QMessageBox.question(self, msg_title, msg,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            event.accept()
            QApplication.quit()
        else:
            event.ignore()

    # TODO analysis functions
    def analysis_detect_edges(self):
        print('Detect edges(Canny) pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='tif', type_analysis='canny')

    def analysis_detect_lines(self):
        print('Detect lines pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='tif', type_analysis='line')

    def analysis_detect_faults(self):
        print('Detect faults pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='faults')

    def analysis_density(self):
        print('Density analysis pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='density', txt='Select linear features SHP file')

    def create_rasterdensmap(self):
        print('Create rasterdensmap pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='rasterdensmap', txt='Select linear features SHP file for raster density GeoTiff')

    def analysis_minkowski(self):
        print('Minkowski analysis pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='minkowski', txt='Select linear features SHP file')

    def analysis_rose_diagram(self):
        print('Rose analysis pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='rose', txt='Select linear features SHP file')

    def analysis_heatmap(self):
        print('Heatmap analysis pressed')
        # open file for analysis select dialogue
        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='heatmap', txt='Select point features SHP file',
                                      gtype='point')

    # TODO analysis_datatable
    def analysis_datatable(self):
        print('Datatable analysis pressed')
        # open file for analysis select dialogue

        self.selectFileForAnyAnalysis(ftype='shp', type_analysis='data', txt='Select line and fault features SHP file',
                                      multichoice=True)
        print('selected files=', self.file_for_analysis)

        # self.selectFileForAnyAnalysis(ftype='shp', type_analysis='data', txt='Select point SHP file',multichoice=False,gtype='point')
        # print('selected files=', self.file_for_analysis)

    # TODO file processing for analysis (selection, prefs)
    def selectFileForAnyAnalysis(self, ftype='tif', type_analysis='line', txt='Select file', gtype='line',
                                 multichoice=False):
        print('Select file for analysis was selected')
        # self.select_file_analysis = selectFileWindow(parent = self)
        self.current_method = type_analysis
        if type_analysis != 'data':
            self.select_file_analysis = selectFileWindow(parent=self, ftype=ftype, gtype=gtype, multichoice=multichoice)
            self.select_file_analysis.show()
        else:
            # ML work
            self.select_file_analysis = selectFileWindowDataTable(parent=self, ftype=ftype, gtype=gtype,
                                                                  multichoice=multichoice)
            self.select_file_analysis.show()

    #TODO do_analysis!!! this !!!function runs any type of analysis from file selection window on Ok
    def do_analysis(self):
        try:
            print(self.current_method, 'method of analysis for', self.file_for_analysis[0], 'has been activated')
        except:
            print('None file was specified')
        # line, faults, density, minkowski
        #TODO store canny settings here from default, update when changed 
        if self.current_method == 'rasterdensmap':
            print('Computing of Raster densmap has to be here!')

            save_geotiff_path = self.save_tif_file()
            if save_geotiff_path:
                
                lines2 = self.get_features_by_name_id(name=self.file_for_analysis[0])
                extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])
                dpxy = [self.settings.value('data')['raster_res'], self.settings.value('data')['raster_res']]
                img = rasterize_shp3(lines2, extent=extent, dpxy=dpxy)
                #img = np.flipud(img) #flip image, correction
                win_size = self.settings.value('data')['raster_radius']

                print('win_size=',win_size)
                hsize, wsize = np.shape(img)

                if win_size > 0.1*hsize or win_size > 0.1*wsize:
                    self.msg_info(self.language_dict['commands']['msg_pixel_size_too_big'][self.selected_language])
                    win_size = int(((hsize+wsize)//2)*0.1)

                rcell, ccell,valcell = [],[],[] #массив координат ячеек окон осреднения и количества точек линий в них

                for cc in range(wsize//win_size + 1):
                    for rr in range(hsize//win_size + 1):
                        sub_matrix = img[rr*win_size:(rr+1)*win_size,cc*win_size:(cc+1)*win_size]
                        rcell.append(rr*win_size+win_size/2)
                        ccell.append(cc*win_size+win_size/2)
                        valcell.append(np.sum(sub_matrix))

                #testing info
                # print('ccell=',ccell)
                # print('rcell=',rcell)
                # print('valcell=',valcell)

                #интерполируем по сетке
                X = np.linspace(min(ccell), max(ccell))
                Y = np.linspace(min(rcell), max(rcell))
                X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
                try:
                    interp = LinearNDInterpolator(list(zip(ccell, rcell)), valcell)
                    img_i = interp(X, Y)
                    #img_i = np.flipud(img_i)
                    hsize, wsize = np.shape(img_i)

                    print('extent=',extent)

                    shp_file_path = self.get_path_by_name_id(name=self.file_for_analysis[0])
                    extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])

                    driver = ogr.GetDriverByName('ESRI Shapefile')
                    dataSource = driver.Open(shp_file_path, 0)  # 0 means read-only. 1 means writeable.
                    layer = dataSource.GetLayer()
                    projection = layer.GetSpatialRef();
                    driver = gdal.GetDriverByName("GTiff")
                    outdata = driver.Create(save_geotiff_path, wsize, hsize, 1, gdal.GDT_Float64)
                    if extent[0]>extent[1]:
                        #dpxy[1] = dpxy[1]*-1
                        horres = -(extent[1]-extent[0])//wsize 
                    else:
                        horres = (extent[1]-extent[0])//wsize 
                    if extent[2]>extent[3]:
                        vertres = -(extent[3]-extent[2])//hsize 
                    else:
                        vertres = (extent[3]-extent[2])//hsize
                    outdata.SetGeoTransform((extent[0], horres, 0, extent[2], 0, vertres));
                    #outdata.SetProjection(projection.ExportToWkt())  ##sets same projection as input
                    outdata.GetRasterBand(1).WriteArray(img_i)
                    outdata.GetRasterBand(1).SetNoDataValue(0)
                    outdata.FlushCache()
                except:
                    self.msg_info(self.language_dict['commands']['msg_unknown_error'][self.selected_language])
                    

                # # reopen file to project
                self.add_data_obj(save_geotiff_path, parent_obj=self.file_for_analysis[0])

                # rewrite data in a layer canny edges (correction)
                try:
                    basename = os.path.basename(save_geotiff_path).split('.')[0]
                except:
                    basename = os.path.basename(save_geotiff_path)
                print('basename=',basename)
                self.set_data_by_name_id(name=basename, newdata=img_i)
                self.update_list2storage()
                del img,img_i 
                
                print('Done!')


        elif self.current_method == 'canny':
            save_geotiff_path = self.save_tif_file(txt='Please, select filename for Canny edges')
            if save_geotiff_path:
                print(save_geotiff_path)
                print('Need to do stadnalone Canny analysis and save GeoTiff')
                print('bin method=',self.settings.value('data')['binarization'])
                print('sigma canny=',self.settings.value('data')['canny_sigma'])
                print('canny_lowt=',self.settings.value('data')['canny_lowt'])
                print('canny_hight=',self.settings.value('data')['canny_hight'])
                try:
                    print('file for analysis',self.file_for_analysis[0])
                except:
                    print('Something is wrong with the file for analysis')
                    return None
                gdal_obj = self.get_gdal_by_name_id(name=self.file_for_analysis[0])
                #print(gdal_obj)

                img = self.get_matrix_by_name_id(name=self.file_for_analysis[0])

                edges_canny_img = adjusted_canny(img,sigma=self.settings.value('data')['canny_sigma'],
                                    low_threshold=self.settings.value('data')['canny_lowt'],
                                    high_threshold=self.settings.value('data')['canny_hight'])

                hsize, wsize = np.shape(edges_canny_img)
                # output to geotiff
                
                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(save_geotiff_path, wsize, hsize, 1, gdal.GDT_UInt16)
                outdata.SetGeoTransform(gdal_obj.GetGeoTransform()) #set geotransform 
                outdata.SetProjection(gdal_obj.GetProjection())#sets projection
                outdata.GetRasterBand(1).WriteArray(edges_canny_img)
                outdata.GetRasterBand(1).SetNoDataValue(-9999) #alpha values
                outdata.FlushCache() ##saves to disk!!
                outdata = None
                edges_canny_img=None
                gdal_obj=None
 
                # reopen file to project
                self.add_data_obj(save_geotiff_path, parent_obj=self.file_for_analysis[0])

                # rewrite data in a layer canny edges (correction)
                try:
                    basename = os.path.basename(save_geotiff_path).split('.')[0]
                except:
                    basename = os.path.basename(save_geotiff_path)
                # print('basename=',basename)
                self.set_data_by_name_id(name=basename, newdata=edges_canny_img)
                self.update_list2storage()
                del edges_canny_img 



        elif self.current_method == 'line':
            # select file name for analysis
            save_shp_path = self.save_shp_file()
            if save_shp_path:
                # TODO add progress bar
                # binary matrix for line detecting
                imBW = gray2binary(self.get_matrix_by_name_id(name=self.file_for_analysis[0]),
                                   method=self.settings.value('data')['binarization'],
                                   sigma=self.settings.value('data')['line_sigma'])

                tres = self.settings.value('data')['line_tresh']
                leng = self.settings.value('data')['line_length']
                lg = self.settings.value('data')['line_gap']

                lines = detectPLineHough2(imBW, tres=tres, leng=leng, lg=lg)

                gdal_obj = self.get_gdal_by_name_id(name=self.file_for_analysis[0])
                # save line to shp
                saveLinesShpFile2(lines, save_shp_path, gdal_obj)
                # reopen file to project
                # TODO specify parent object
                self.add_data_obj(save_shp_path, parent_obj=self.file_for_analysis[0], content='line')

        # TODO heatmap
        elif self.current_method == 'heatmap':
            # extent of selected point shp file
            print(self.file_for_analysis[0])
            print(self.get_extent_by_name_id(name=self.file_for_analysis[0]))
            # select file name for analysis
            save_geotiff_path = self.save_tif_file()
            if save_geotiff_path:
                print(save_geotiff_path)
                point_file_name = self.file_for_analysis[0]
                raster_pixel_size = [self.settings.value('data')['raster_res'],self.settings.value('data')['raster_res']]
                raster_radius = self.settings.value('data')['raster_radius']
                
                shp_file_path = self.get_path_by_name_id(name=self.file_for_analysis[0])
                extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])
                heatmap = np.flipud(self.generate_heatmap(point_file_name, raster_pixel_size, raster_radius, aoi_extent=extent))
                # points = self.get_features_by_name_id(name=self.file_for_analysis[0])
                # extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])
                # shp_file_path = self.get_path_by_name_id(name=self.file_for_analysis[0])
                # #TODO add extent in two line below
                # xlist = [pnt[0][0] for pnt in points]
                # ylist = [pnt[0][1] for pnt in points]
                # x = [xval - min(xlist) for xval in xlist]
                # y = [yval - min(ylist) for yval in ylist]
                # xminmax, yminmax = [min(x), max(y)], [min(x), max(y)]
                #
                # # DEFINE GRID SIZE AND RADIUS(h)
                # # grid_size = 100
                # # h = grid_size * 100
                # raster_pixel_size = self.settings.value('data')['raster_res']
                # raster_radius = self.settings.value('data')['raster_radius']
                #
                # heatmap = get_probability_matrix(x,y,xminmax,yminmax,grid_size=raster_pixel_size,h=raster_radius)
                hsize, wsize = np.shape(heatmap)
                # output to geotiff
                # saveGeoTiff(obj_raster,raster_path,gdal_object,ColMinInd,RowMinInd);
                driver = ogr.GetDriverByName('ESRI Shapefile')
                dataSource = driver.Open(shp_file_path, 0)  # 0 means read-only. 1 means writeable.
                # количество объектов в слое
                layer = dataSource.GetLayer()
                projection = layer.GetSpatialRef();
                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(save_geotiff_path, wsize, hsize, 1, gdal.GDT_Float64)
                outdata.SetGeoTransform((extent[0], raster_pixel_size[0], 0, extent[2], 0, raster_pixel_size[1]));
                outdata.SetProjection(projection.ExportToWkt())  ##sets same projection as input
                outdata.GetRasterBand(1).WriteArray(heatmap)
                outdata.GetRasterBand(1).SetNoDataValue(0)
                outdata.FlushCache()

                # reopen file to project
                # self.add_data_obj(save_geotiff_path,parent_obj=self.file_for_analysis)
                self.add_data_obj(save_geotiff_path, parent_obj=self.file_for_analysis[0])

                # rewrite data in a layer heatmap (correction)
                try:
                    basename = os.path.basename(save_geotiff_path).split('.')[0]
                except:
                    basename = os.path.basename(save_geotiff_path)
                # print('basename=',basename)
                self.set_data_by_name_id(name=basename, newdata=heatmap)
                self.update_list2storage()
                del heatmap  # remove heatmap matrix


        elif self.current_method == 'rose':
            txt = self.language_dict['commands']['rose_label'][self.selected_language]
            new_lines = self.get_features_by_name_id(name=self.file_for_analysis[0])
            build_rose_diag(new_lines, txt=txt + ' ' + self.file_for_analysis[0])

        elif self.current_method == 'faults':
            # TODO fault detection over rasterized lines
            print('not done yet. need to operate over rasterized image')
            save_shp_path = self.save_shp_file()
            if save_shp_path:
                tres = self.settings.value('data')['fault_tresh']
                leng = self.settings.value('data')['fault_length']
                lg = self.settings.value('data')['fault_gap']

                parent_obj = self.get_parent_by_name_id(name=self.file_for_analysis[0])
                gdal_obj = self.get_gdal_by_name_id(name=parent_obj)
                new_lines = self.get_features_by_name_id(name=self.file_for_analysis[0])

                # rasterize image
                # print('lines',lines)
                # print('gdal_object',gdal_obj)
                if gdal_obj:
                    imBW = rasterize_shp(new_lines, gdal_object=gdal_obj)
                    ext = None
                    dpxy = None
                else:
                    lines2 = self.get_features_by_name_id(name=self.file_for_analysis[0])
                    extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])
                    dpxy = [self.settings.value('data')['raster_res'], self.settings.value('data')['raster_res']]
                    imBW = rasterize_shp3(lines2, extent=extent, dpxy=dpxy)
                    # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
                    ext = [[extent[0], extent[3]], [extent[0], extent[2]], [extent[1], extent[2]],
                           [extent[1], extent[3]]]
                # test raster save
                # TODO saves tif only if parent is known
                if self.get_parent_by_name_id(name=self.file_for_analysis[0]):
                    save_tif_path = save_shp_path.replace('.shp', '.tif')
                    saveGeoTiff(imBW, save_tif_path, gdal_obj)

                # detect faults
                faults = detectPLineHough2(imBW, tres=tres, leng=leng, lg=lg)
                # save line to shp
                saveLinesShpFile2(faults, save_shp_path, gdal_object=gdal_obj, ext=ext, dpxy=dpxy)
                # reopen file to project
                self.add_data_obj(save_shp_path, parent_obj=parent_obj, content='faults')


        elif self.current_method == 'density' or self.current_method == 'minkowski':
            save_shp_path = self.save_shp_file()
            if save_shp_path:
                if self.current_method == 'density':
                    ws = self.settings.value('data')['density_window_size']
                elif self.current_method == 'minkowski':
                    ws = self.settings.value('data')['fractal_window_size']
                else:
                    ws = None
                # analysis, get data_dict
                # self.select_file_analysis
                lines2 = self.get_features_by_name_id(name=self.file_for_analysis[0])
                extent = self.get_extent_by_name_id(name=self.file_for_analysis[0])
                dpxy = [self.settings.value('data')['raster_res'], self.settings.value('data')['raster_res']]
                img = rasterize_shp3(lines2, extent=extent, dpxy=dpxy)
                # gdal_obj = self.get_gdal_by_name_id(name=self.select_file_analysis)
                # img = self.get_matrix_by_name_id(name=self.file_for_analysis)

                if self.current_method == 'density':
                    data_dict = get_pixels_sum(img, win_size=ws,
                                               spec_field=self.current_method,
                                               gdal_obj=None, extent=extent, dpxy=dpxy)
                elif self.current_method == 'minkowski':
                    data_dict = get_minkowski(img, win_size=ws,
                                              spec_field=self.current_method,
                                              gdal_obj=None, extent=extent, dpxy=dpxy)
                else:
                    data_dict = None

                print(data_dict)
                createSHPfromDictionary(save_shp_path, data_dict)

        # TODO implement target variables computing and data table doing here
        elif self.current_method == 'data':
            csv_path = QFileDialog.getSaveFileName(self, ("Save datatable as"), '', ("csv file (*.csv)"))[0]
            if csv_path != '':
                print('start of data method')
                # dictionary for values per layer
                img_dict = {}
                minkowski_dict = {}
                density_dict = {}

                # collect lineament and faults data
                x_margins_layers = []
                y_margins_layers = []

                all_files = self.file_for_analysis + [self.file_for_analysis_target]

                for file in all_files:  # self.file_for_analysis - SEVERAL SELECTED FILENAMES
                    print(file)
                    # get data object for file with given name
                    obj = self.get_obj_by_name_id(name=file)
                    # get extent
                    x_margins_layers.append(obj.data['extent'][0])
                    x_margins_layers.append(obj.data['extent'][1])
                    y_margins_layers.append(obj.data['extent'][2])
                    y_margins_layers.append(obj.data['extent'][3])

                # compute extent and resolution for selected layers
                aoi_extent = (min(x_margins_layers), max(x_margins_layers),
                              min(y_margins_layers), max(y_margins_layers))
                dpxy = [self.settings.value('data')['raster_res'], self.settings.value('data')['raster_res']]
                # dpxy = self.settings.value('data')['raster_res']

                #test
                test_width = int((aoi_extent[1] - aoi_extent[0]) / dpxy[0])
                test_heigth = int((aoi_extent[3] - aoi_extent[2]) / dpxy[1])
                #test
                print('test_width=',test_width)
                print('test_heigth=',test_heigth)
                print('aoi_extent=',aoi_extent)


                for file in self.file_for_analysis:
                    line_faults = self.get_features_by_name_id(name=file)
                    img = rasterize_shp3(line_faults, extent=aoi_extent, dpxy=dpxy)
                    img_dict.update({file: img})

                # data for analysis by image type
                ws = min([self.settings.value('data')['density_window_size'],
                          self.settings.value('data')['fractal_window_size']])
                for file in self.file_for_analysis:
                    minkowski_dict.update({file: get_minkowski(img_dict[file], win_size=ws,
                                                               spec_field=file + '_minkowski',
                                                               gdal_obj=None, extent=aoi_extent, dpxy=dpxy)})

                    density_dict.update({file: get_pixels_sum(img_dict[file], win_size=ws,
                                                              spec_field=file + '_density',
                                                              gdal_obj=None, extent=aoi_extent, dpxy=dpxy)})

                # target points probability density
                raster_radius = self.settings.value('data')['raster_radius']
                density_prob_image = self.generate_heatmap(self.file_for_analysis_target, dpxy, raster_radius,
                                                           aoi_extent=aoi_extent)

                target_points_density = get_average_val(density_prob_image, win_size=ws,
                                                        spec_field=self.file_for_analysis_target + '_pop',
                                                        gdal_obj=None, extent=aoi_extent, dpxy=dpxy)

                # arrange output table
                # TODO assemble output data table
                for field in [*target_points_density]:
                    print(field, '=', target_points_density[field])
                print('===============')
                for field in [*density_dict]:
                    print(field, '=', density_dict[field])
                print('===============')
                for field in [*minkowski_dict]:
                    print(field, '=', minkowski_dict[field])

                print('len of target_points_density=', len(target_points_density['id']))
                for file in self.file_for_analysis:
                    print(f'len of mink {file}= {len(minkowski_dict[file]["id"])}')
                    print(f'len of dens {file}= {len(density_dict[file]["id"])}')

                # output image resolutions
                for key in self.file_for_analysis:
                    print('key=', key, 'resolution=', np.shape(img_dict[key]))

                print('key=probability', 'resolution=', np.shape(density_prob_image))

                # minkowski_dict
                # density_dict
                # target_points_density

                #dictionary output
                data_out = copy.deepcopy(target_points_density)
                add_dicts = [density_dict,minkowski_dict]
                for dictionary in add_dicts:
                    for field in [*dictionary]:
                        if field not in [*data_out]:
                            data_out.update(dictionary[field])

                print('resulting df')
                dataframe_out = pd.DataFrame.from_dict(data_out)
                dataframe_out.to_csv(csv_path, index=False)


        else:
            self.msg_info('No analysis methods were specified')
            return None

    # TODO define structure from opened tiff data (filetype by extension)
    def add_data_obj(self, filename, parent_obj=None, content=None):
        """
        Структура данных массива открытых растров как список словарей:
        Data structure of opened raster arrays as dictionary list
        {'id':____} unique integer number autoincrement
        {'name':____} filename without extension
        {'file_path':____} file path full
        {'type':____} tif shp
        {'parent':____} none if opened from disk or created from no sources, otherwise contains integer ID of parent file
        {'gdal_obj':____} gdal object for georef data, none if nogeoref
        {'draw_order':____} integer for obj drawing order
        """
        # add object to data storage
        self.data_storage.open_data(filename, parent_obj=parent_obj, content=content)

        # update layers list
        self.update_list2storage()

    # TODO remove from layers (project file) list
    def layers_list_remove(self, name: str):
        if self.msg_yn(msg=self.language_dict['commands']['msg_yn_text'][self.selected_language]) == 'y':
            item_list = [item.text() for item in self.layers.layer_list.selectedItems()]
            for name in item_list:
                for num, obj in enumerate(self.data_storage.get_objects()):
                    if obj.name == name:
                        self.data_storage.list_objects.pop(num)
                        # close file browser
                        self.map_browser.close()
                        self.update_list2storage()
                        break

    def show_stat(self):
        print('show stat was called')
        try:
            item_list = [item.text() for item in self.layers.layer_list.selectedItems()]
            msg_info_text = ''
            # iterate selected layers and compose output
            for item_name in item_list:
                msg_info_text = msg_info_text + 'FILENAME: ' + item_name + '\n'
                obj = self.get_obj_by_name_id(self, name=item_name)
                for key in [*obj.__dict__]:
                    msg_info_text = msg_info_text + key + ':' + str(eval('obj.' + key)) + '\n'
                    if key == 'data':
                        msg_info_text = msg_info_text + 'features count' + ':'+str(len(obj.data['features']))+ '\n'




            self.showStatWin(msg=msg_info_text, title=item_list[0])
            print(msg_info_text)
            # self.msg_info(msg=msg_info_text)

        except IndexError:
            print('Layer list seems empty')

    # TODO def layers to xml project
    def datastorage2xml(self, path):
        xmlhead = '''<?xml version="1.0" encoding="UTF-8"?><body>'''
        xmlfoot = '''</body>'''
        xml2out = '' + xmlhead
        for obj in self.data_storage.get_objects():
            objxml = f'''
            <obj{obj.id}><id>{obj.id}</id><file_path>{obj.file_path}</file_path>
            <type>{obj.type}</type>
            <parent>{obj.parent_obj}</parent>
            <content>{obj.content}</content>
            <draw_order>{obj.draw_order}</draw_order><name>{obj.name}</name>
            </obj{obj.id}>
            '''

            xml2out = xml2out + objxml
        xml2out = xml2out + xmlfoot
        try:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(xml2out)
        except:
            QMessageBox.critical(self, 'Error saving project file.',
                                 'Project file could not be saved!',
                                 QMessageBox.Ok, QMessageBox.Ok)

    # TODO read xml to project
    def xml2project(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                my_xml = file.read()
            my_dict = xmltodict.parse(my_xml)
            print(my_dict)
            return my_dict
        except:
            QMessageBox.critical(self, 'Error loading project file.',
                                 'Project file could not be loaded!',
                                 QMessageBox.Ok, QMessageBox.Ok)
            return None

    # TODO get data MATRIX by name or id
    def get_matrix_by_name_id(self, id=None, name=None):
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                if obj.type == 'tif':
                    if obj.gdal_obj == None and obj.band == None:
                        print('No matrix data were found')
                        return None
                    elif obj.gdal_obj == None and obj.band != None:
                        return obj.band
                    elif obj.gdal_obj != None:
                        return obj.get_band_from_gdal(obj.gdal_obj)
                    else:
                        print('No band data were found')
                        return None
                elif obj.type == 'shp':
                    return obj.data
                else:
                    print('unknown format')
                    return None

    def get_parent_by_name_id(self, id=None, name=None):
        print('get parent')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                print(obj.parent_obj)
                return obj.parent_obj

    # TODO get data MATRIX by name or id
    def get_type_by_name_id(self, id=None, name=None):
        print('get type')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                print(obj.type)
                return obj.type

    # TODO get data MATRIX by name or id
    def get_content_by_name_id(self, id=None, name=None):
        print('get content fault/line')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                print(obj.content)
                return obj.content

    # TODO get data MATRIX by name or id
    def get_gdal_by_name_id(self, id=None, name=None):
        print('get gdal')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                return obj.gdal_obj

    def get_obj_by_name_id(self, id=None, name=None):
        print('get obj')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                try:
                    return copy.deepcopy(obj)
                except TypeError:
                    print('can not deepcopy an obj')
                    return copy.copy(obj)

    def get_features_by_name_id(self, id=None, name=None):
        print('get features')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                return copy.deepcopy(obj.data['features'])

    def set_data_by_name_id(self, id=None, name=None, newdata=None):
        print('set data')
        if id == None and name == None:
            print('No name or id was given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                print('object was found')
                obj.data = newdata
                self.data_storage.delete_obj_byname(name=name)
                self.data_storage.add_obj(obj)
                break

    def get_extent_by_name_id(self, id=None, name=None):
        print('get extent')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                return copy.deepcopy(obj.data['extent'])

    def get_path_by_name_id(self, id=None, name=None):
        print('get path')
        if id == None and name == None:
            print('No data for selecting were given')
            return None
        obj_list = self.data_storage.get_objects()
        for obj in obj_list:
            if (obj.id == id and obj.id != None) or (obj.name == name and obj.name != None):
                return copy.deepcopy(obj.file_path)

    # TODO update layer list
    def update_list2storage(self):
        self.layers.layer_list.clear()  # clear all items
        for obj in self.data_storage.get_objects():
            # add to list items
            item1 = QListWidgetItem()  # need to copy theese items twice
            item1.setText(obj.name)
            # item1.setToolTip(self.language_dict['commands']['map_window_title_srtm'][self.selected_language])
            item1.setToolTip(obj.file_path)

            # setting icon depending on content type
            icon = QIcon()
            if obj.type == 'tif':
                path = self.pict_rast_layer
            elif obj.type == 'shp':
                if obj.data['type'] != 'point':
                    path = self.pict_vect_layer
                else:
                    path = self.pict_pnt_layer
            else:
                path = self.pict_unk_layer

            icon.addPixmap(QPixmap(path), QIcon.Normal, QtGui.QIcon.Off)
            item1.setIcon(icon)

            # TODO when open map browser?
            # self.map_browser.setWindowTitle(
            #    self.language_dict['commands']['map_window_title_srtm'][self.selected_language])

            # self.list1.addItem(item1)
            self.layers.layer_list.addItem(item1)
            self.layers.layer_list.setSelectionMode(QAbstractItemView.SingleSelection)
            # self.layers.layer_list.itemSelectionChanged.connect(self.on_change1)
            # self.layers.layer_list.itemClicked.connect(self.on_change1)

    # TODO def file_open_dialogue OPEN FILE
    def file_open_dialogue(self, type_filter='tif'):
        msg = self.language_dict['commands']['app_dialog_open'][self.selected_language]
        print('type_filter=', type_filter)
        if type_filter == 'tif':
            name_mask = "Tiff (*.tif *.tiff);; ESRI Shape (*.shp)"
        elif type_filter == 'shp':
            name_mask = "ESRI Shape (*.shp)"
        else:
            name_mask = "All files (*.*)"

        fileNames = QFileDialog.getOpenFileNames(self, ("Open File"), '', name_mask)
        if fileNames != '':
            print(fileNames[0])
            for fileName in fileNames[0]:
                # TODO add data object opened from disk
                self.add_data_obj(fileName)

    def about_dialogue(self):
        txt_label = self.language_dict['commands']['app_info_label'][self.selected_language]
        txt_info = self.language_dict['commands']['app_info_title'][self.selected_language]

        print(txt_label)
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setTextFormat(Qt.RichText)
        msgBox.setText(
            txt_label + '\n <br><a href=\'http://fegi.ru\'>fegi.ru</a> <br> <a href=\'http://lefa.geologov.net\'>lefa.geologov.net</a> <br> <a href=\'https://t.me/pylefa\'>telegram</a>')
        msgBox.exec_()

    # TODO show result method
    def show_result(self, img_arr, title_arr, title=['Map browser'], parent_obj=None, content=None):  # im_arr in []
        self.map_browser.figure.clear()
        ax = self.map_browser.figure.add_subplot(111)
        ax.cla()  # or picture will not be updated
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        print('show picture')

        if 'arr' in str(type(img_arr[0])):  # if tif numpy array type
            if self.output == 'contour':
                # im1 = ax.contour(img_arr[0], cmap='terrain', interpolation='nearest')
                im1 = ax.contour(img_arr[0], cmap="viridis")
                # levels=list(range(0, 5000, 100)))
                ax.set_aspect('equal', adjustable='box')
                ax.invert_yaxis()
                try:
                    self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)
                except:
                    print('Can not add colorbar to contours')

            elif self.output == 'raster':
                # print(type(img_arr[0]))
                # print(type(type(img_arr[0])))
                # print('try to show raster')
                im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)

        elif 'dict' in str(type(img_arr[0])):
            data = img_arr[0]
            ext = data['extent']
            features = data['features']
            print('objects totally:', len(features))

            if parent_obj[0] != None:
                gdal_object1 = self.get_gdal_by_name_id(name=parent_obj[0])
                # detect parent resolution
                try:
                    gt1 = gdal_object1.GetGeoTransform()
                    cols1 = gdal_object1.RasterXSize
                    rows1 = gdal_object1.RasterYSize
                    ext1 = GetExtent(gt1, cols1, rows1)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
                except AttributeError:  # parent was deleted from project
                    ext1 = copy.copy(ext)

            #
            if content[0] != None:
                cont_line = self.get_content_by_name_id(name=title_arr[0])
            else:
                cont_line = None

            # colors for lines on a plot
            if cont_line == 'line':
                lc = 'b-'
            elif cont_line == 'faults':
                lc = 'r-'
            else:
                lc = 'c-'

            id_count = 0
            pbar_window = ProgressBar()
            print('check ')
            if str(parent_obj[0]) != 'None':
                ax.imshow(self.get_matrix_by_name_id(name=parent_obj[0]))
            for feat in features:
                pbar_window.doProgress(id_count, len(features))
                x = []
                y = []

                # show background picture if there is a parent_obj
                if str(parent_obj[0]) != 'None':
                    try:
                        ax.imshow(self.get_matrix_by_name_id(name=parent_obj[0]))
                    except:
                        print('can not show bgrd image')

                for pnt in feat:
                    x.append(pnt[0] - ext[0])
                    if str(parent_obj[0]) != 'None':
                        y.append(ext1[0][1] - pnt[1])
                    else:
                        y.append(pnt[1] - ext[2])

                if str(parent_obj[0]) != 'None':
                    # resolution in meters
                    dpx = (ext1[3][0] - ext1[0][0]) / cols1
                    dpy = (ext1[0][1] - ext1[2][1]) / rows1
                    x = [px / dpx for px in x]
                    y = [py / dpy for py in y]
                    # print('x=',x)
                    # print('y=',y)
                    if data['type'] != 'point':
                        im1 = ax.plot(x, y, lc)
                    else:
                        im1 = ax.plot(x[0], y[0], 'bo')
                else:
                    # print('x=', x)
                    # print('y=', y)
                    if data['type'] != 'point':
                        im1 = ax.plot(x, y, lc)
                    else:
                        im1 = ax.plot(x[0], y[0], 'bo')

                id_count += 1


        else:
            print('show raster alone')
            im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
            self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)

        self.map_browser.setWindowTitle(','.join(title))
        self.map_browser.canvas.draw()
        self.map_browser.show()
        self.map_browser.raise_()

    def show_result_old(self, img_arr, title_arr, title=['Map browser'], parent_obj=None, content=None):  # im_arr in []
        print(self.output)
        self.map_browser.figure.clear()
        ax = self.map_browser.figure.add_subplot(111)
        ax.cla()  # or picture will not be updated
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        print('len(img_arr)', len(img_arr))
        if len(img_arr) != 1:  # WE DONT USE IT, TWO or more layers to show!
            im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
            im2 = ax.imshow(img_arr[1], cmap='terrain', alpha=.50, interpolation='nearest')
            self.map_browser.figure.colorbar(im2, ax=ax, orientation='vertical', cax=cax)
            # im3 = ax.imshow(img_arr[2], cmap=plt.cm.viridis, alpha=.95, interpolation='bilinear')
        else:  # если вывод просто картинки, i.e. ONLY one layer to SHOW!
            print('show picture')

            if 'arr' in str(type(img_arr[0])):  # if tif numpy array type
                if self.output == 'contour':
                    # im1 = ax.contour(img_arr[0], cmap='terrain', interpolation='nearest')
                    im1 = ax.contour(img_arr[0], cmap="viridis")
                    # levels=list(range(0, 5000, 100)))
                    ax.set_aspect('equal', adjustable='box')
                    ax.invert_yaxis()
                    try:
                        self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)
                    except:
                        print('Can not add colorbar to contours')

                elif self.output == 'raster':
                    print(type(img_arr[0]))
                    print(type(type(img_arr[0])))
                    print('try to show raster')
                    im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                    self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)

            elif 'dict' in str(type(img_arr[0])):
                # pbar_window = ProgressBar()
                # print('possibly shape shp file?')
                # determine color of line to be shown

                data = img_arr[0]
                ext = data['extent']
                # print(ext)
                features = data['features']
                # print(features)
                print('objects totally:', len(features))

                if parent_obj[0] != None:
                    gdal_object1 = self.get_gdal_by_name_id(name=parent_obj[0])
                    # detect parent resolution
                    try:
                        gt1 = gdal_object1.GetGeoTransform()
                        cols1 = gdal_object1.RasterXSize
                        rows1 = gdal_object1.RasterYSize
                        ext1 = GetExtent(gt1, cols1, rows1)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
                    except AttributeError:  # parent was deleted from project
                        ext1 = copy.copy(ext)

                #
                if content[0] != None:
                    cont_line = self.get_content_by_name_id(name=title_arr[0])
                else:
                    cont_line = None

                # colors for lines on a plot
                if cont_line == 'line':
                    lc = 'b-'
                elif cont_line == 'faults':
                    lc = 'r-'
                else:
                    lc = 'c-'

                id_count = 0
                print('id_count', id_count)
                pbar_window = ProgressBar()
                if parent_obj[0] != None:
                    ax.imshow(self.get_matrix_by_name_id(name=parent_obj[0]))
                for feat in features:
                    pbar_window.doProgress(id_count, len(features))
                    x = []
                    y = []
                    # print('ext[0]:',ext[0])
                    # print('ext[2]:',ext[2])

                    # show background picture if there is a parent_obj
                    if parent_obj[0] != None:
                        try:
                            ax.imshow(self.get_matrix_by_name_id(name=parent_obj[0]))
                        except:
                            print('can not show bgrd image')

                    for pnt in feat:
                        x.append(pnt[0] - ext[0])
                        if parent_obj[0] != None:
                            # y.append(ext[3] - pnt[1])
                            y.append(ext1[0][1] - pnt[1])
                        else:
                            # y.append(ext[3] - pnt[1])
                            y.append(pnt[1] - ext[2])

                    # show background picture if there is a parent_obj
                    # try:
                    #     ax.imshow(self.get_matrix_by_name_id(name=parent_obj[0]))
                    # except:
                    #     print('can not show bgrd image')
                    # try:
                    if parent_obj[0] != None:
                        # print('parent_obj==',parent_obj[0])
                        # TODO ax.imshow()
                        # try to output bgrd

                        # ax.invert_yaxis()
                        # resolution in meters
                        dpx = (ext1[3][0] - ext1[0][0]) / cols1
                        dpy = (ext1[0][1] - ext1[2][1]) / rows1
                        x = [px / dpx for px in x]
                        y = [py / dpy for py in y]
                        im1 = ax.plot(x, y, lc)

                    else:
                        im1 = ax.plot(x, y, lc)
                        # ax.invert_yaxis()

                    # except:
                    # print('parent obj is inaccessible')
                    # im1 = ax.plot(x, y, lc)
                    # ax.invert_yaxis()

                    id_count += 1

            else:
                print('show raster alone')
                im1 = ax.imshow(img_arr[0], cmap='terrain', interpolation='nearest')
                self.map_browser.figure.colorbar(im1, ax=ax, orientation='vertical', cax=cax)

        self.map_browser.setWindowTitle(','.join(title))
        self.map_browser.canvas.draw()
        self.map_browser.show()
        self.map_browser.raise_()

    def on_change1(self):
        print('on_change1 was called')
        # TODO self.list1.selectedItems()
        item_list = [item.text() for item in self.layers.layer_list.selectedItems()]
        if len(item_list) != 0:
            mat_list = [self.get_matrix_by_name_id(name=lname) for lname in item_list]
            parent_list = [self.get_parent_by_name_id(name=lname) for lname in item_list]
            content_list = [self.get_content_by_name_id(name=lname) for lname in item_list]
            self.show_result(mat_list, item_list, parent_obj=parent_list, content=content_list)


class StatWindow(QWidget):
    def __init__(self, parent=None, title='', txt=''):
        super().__init__()
        self.parent = parent
        self.txt_field = QPlainTextEdit(self)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.txt_field)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.setMinimumSize(QSize(440, 280))
        self.setWindowTitle('info: ' + title)
        self.txt_field.setPlainText(txt)
        self.show()


# TODO WINDOW class for selecting file for analysis
class selectFileWindow(QWidget):
    def __init__(self, parent=None, ftype='tif', gtype='line', multichoice=False):
        super().__init__()
        self.parent = parent
        txt_title = self.parent.language_dict['commands']['select_file_analysis_title'][self.parent.selected_language]

        self.setWindowTitle(txt_title)
        self.list1 = QListWidget()

        txt_label = self.parent.language_dict['commands']['select_file_analysis_label'][self.parent.selected_language]
        self.label1 = QLabel(txt_label)

        self.OKbnt = QPushButton(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])
        self.CancelBnt = QPushButton(self.parent.language_dict['commands']['cancel_btn'][self.parent.selected_language])
        self.PrefBnt = QPushButton(self.parent.language_dict['commands']['pref_btn'][self.parent.selected_language])

        # populate list
        self.list1.clear()  # clear all items
        # add to list items
        print('getting objs from storage')
        print(self.parent.data_storage.get_objects())
        for obj in self.parent.data_storage.get_objects():
            if obj.type == ftype:

                # TODO select POINTS for certain type of analysis
                if obj.type == 'shp':
                    if obj.data['type'] != gtype:
                        # print('obj.data[type]=',obj.data['type'])
                        # print('gtype=',gtype)
                        continue  # ignore shp without default geometry

                # add to list items
                item1 = QListWidgetItem()  # need to copy theese items twice
                print(obj.name)
                item1.setText(obj.name)
                item1.setToolTip(obj.file_path)

                self.list1.addItem(item1)
        # self.list1.setSelectionMode(QAbstractItemView.SingleSelection)
        if multichoice == True:
            self.list1.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.list1.setSelectionMode(QAbstractItemView.SingleSelection)

        vbox = QVBoxLayout()
        vbox.addWidget(self.list1)
        vbox.addWidget(self.label1)
        vbox.addWidget(self.PrefBnt)
        vbox.addWidget(self.CancelBnt)
        vbox.addWidget(self.OKbnt)

        self.setLayout(vbox)
        self.resize(300, 150)

        self.CancelBnt.clicked.connect(self.on_cancel)
        self.PrefBnt.clicked.connect(self.parent.showPreferences)
        self.OKbnt.clicked.connect(self.on_ok)

    def on_cancel(self):
        self.close()

    def on_ok(self):
        print('ok was pressed')
        try:
            item_text = [item.text() for item in self.list1.selectedItems()]
        except:
            self.parent.msg_info(
                self.parent.language_dict['commands']['msg_specify_file'][self.parent.selected_language])
            return None
        txt_question = self.parent.language_dict['commands']['select_file_dialog'][self.parent.selected_language]
        txt_title = self.parent.language_dict['commands']['select_file_dialog_title'][self.parent.selected_language]
        msgBox = QMessageBox.question(self, txt_title, txt_question.replace('#', str(item_text)),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            # add layer to file processing var
            self.parent.file_for_analysis = item_text
            print(item_text)
        self.close()
        # launching universal function, that perform any analysis
        # based on !!!flags and parameters
        self.parent.do_analysis()
        return None


# class for data analysis
class selectFileWindowDataTable(QWidget):
    def __init__(self, parent=None, ftype='tif', gtype='line', multichoice=False):
        super().__init__()
        self.parent = parent
        txt_title = self.parent.language_dict['commands']['select_file_analysis_title'][self.parent.selected_language]

        self.setWindowTitle(txt_title)
        self.list1 = QListWidget()
        self.list2 = QListWidget()

        txt_label = self.parent.language_dict['commands']['select_pred_file_analysis_label'][
            self.parent.selected_language]
        txt_label2 = self.parent.language_dict['commands']['select_target_file_analysis_label'][
            self.parent.selected_language]
        self.label1 = QLabel(txt_label)
        self.label2 = QLabel(txt_label2)

        self.OKbnt = QPushButton(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])
        self.CancelBnt = QPushButton(self.parent.language_dict['commands']['cancel_btn'][self.parent.selected_language])
        self.PrefBnt = QPushButton(self.parent.language_dict['commands']['pref_btn'][self.parent.selected_language])

        # populate list
        self.list1.clear()  # clear all items
        self.list2.clear()  # clear all items
        # add to list items
        print('getting predictors from storage')
        print(self.parent.data_storage.get_objects())
        for obj in self.parent.data_storage.get_objects():
            if obj.type == ftype:

                # TODO select POINTS for certain type of analysis
                if obj.type == 'shp':
                    if obj.data['type'] != gtype:
                        # print('obj.data[type]=',obj.data['type'])
                        # print('gtype=',gtype)
                        continue  # ignore shp without default geometry

                # add to list items
                item1 = QListWidgetItem()  # need to copy theese items twice
                print(obj.name)
                item1.setText(obj.name)
                item1.setToolTip(obj.file_path)

                self.list1.addItem(item1)

        for obj in self.parent.data_storage.get_objects():
            if obj.type == ftype:

                # TODO select POINTS for certain type of analysis
                if obj.type == 'shp':
                    if obj.data['type'] != 'point':
                        continue  # ignore shp without default geometry

                # add to list items
                item1 = QListWidgetItem()  # need to copy theese items twice
                print(obj.name)
                item1.setText(obj.name)
                item1.setToolTip(obj.file_path)

                self.list2.addItem(item1)

        if multichoice == True:
            self.list1.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.list1.setSelectionMode(QAbstractItemView.SingleSelection)

        self.list2.setSelectionMode(QAbstractItemView.SingleSelection)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label1)
        vbox.addWidget(self.list1)
        vbox.addWidget(self.label2)
        vbox.addWidget(self.list2)

        vbox.addWidget(self.PrefBnt)
        vbox.addWidget(self.CancelBnt)
        vbox.addWidget(self.OKbnt)

        self.setLayout(vbox)
        self.resize(300, 150)

        self.CancelBnt.clicked.connect(self.on_cancel)
        self.PrefBnt.clicked.connect(self.parent.showPreferences)
        self.OKbnt.clicked.connect(self.on_ok)

    def on_cancel(self):
        self.close()

    def on_ok(self):
        print('ok was pressed')
        try:
            item_text = [item.text() for item in self.list1.selectedItems()]
        except:
            self.parent.msg_info(
                self.parent.language_dict['commands']['msg_specify_file'][self.parent.selected_language])
            return None

        try:
            target_item = self.list2.selectedItems()[0].text()
        except:
            self.parent.msg_info(
                self.parent.language_dict['commands']['msg_specify_file'][self.parent.selected_language])
            return None

        txt_question = self.parent.language_dict['commands']['select_file_dialog'][self.parent.selected_language]
        txt_title = self.parent.language_dict['commands']['select_file_dialog_title'][self.parent.selected_language]
        msgBox = QMessageBox.question(self, txt_title, txt_question.replace('#', str(item_text)),
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if msgBox == QMessageBox.Yes:
            # add layer to file processing var
            self.parent.file_for_analysis = item_text
            self.parent.file_for_analysis_target = target_item
            print(item_text)
            print(target_item)
        self.close()
        # launching universal function, that perform any analysis
        # based on !!!flags and parameters
        self.parent.do_analysis()
        return None


# TODO add user preferences save and load
class Preferences(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.setWindowTitle(self.parent.language_dict['commands']['app_dialog_pref'][self.parent.selected_language])

        # elements for line detecting analysis
        # global window elements
        btnOk = QPushButton(self.parent.language_dict['commands']['ok_btn'][self.parent.selected_language])
        btnCancel = QPushButton(self.parent.language_dict['commands']['cancel_btn'][self.parent.selected_language])
        btnOk.clicked.connect(self.applySettings)
        btnCancel.clicked.connect(self.close)

        # line analysis widgets
        tab_line_name = self.parent.language_dict['commands']['line_det_tab_name'][self.parent.selected_language]
        self.radioNone = QRadioButton(
            self.parent.language_dict['commands']['line_det_bin_none'][self.parent.selected_language])
        self.radioCanny = QRadioButton(
            self.parent.language_dict['commands']['line_det_bin_canny'][self.parent.selected_language])
        self.radioFlows = QRadioButton(
            self.parent.language_dict['commands']['line_det_bin_flows'][self.parent.selected_language])
        # sigma for Canny analysis parameters
        self.sigmaCannyLabel = QLabel(
            self.parent.language_dict['commands']['line_det_bin_canny_sigma'][self.parent.selected_language])
        self.sigmaCannySpin = QSpinBox(self)
        self.sigmaCannySpin.setRange(1, 50)
        self.sigmaCannySpin.setValue(self.parent.settings.value('data')['line_sigma'])
        self.sigmaCannySpin.setEnabled(False)

        if self.parent.settings.value('data')['binarization'] == 'canny':
            self.radioCanny.setChecked(True)
            self.sigmaCannySpin.setEnabled(True)
        elif self.parent.settings.value('data')['binarization'] == 'none':
            self.radioNone.setChecked(True)
            self.sigmaCannySpin.setEnabled(False)
        else:
            self.radioFlows.setChecked(True)
            self.sigmaCannySpin.setEnabled(False)
        binaryMethodLabel = QLabel(
            self.parent.language_dict['commands']['line_det_bin_label'][self.parent.selected_language])
        houghTransformLabel = QLabel(
            self.parent.language_dict['commands']['line_det_hough_label'][self.parent.selected_language])
        houghTransformTreshLabel = QLabel(
            self.parent.language_dict['commands']['line_det_treshold_label'][self.parent.selected_language])
        houghTransformLenLabel = QLabel(
            self.parent.language_dict['commands']['line_det_len_label'][self.parent.selected_language])
        houghTransformGapLabel = QLabel(
            self.parent.language_dict['commands']['line_det_gap_label'][self.parent.selected_language])
        self.spinHoughTresh = QSpinBox(self)
        self.spinHoughTresh.setRange(5, 100)
        self.spinHoughTresh.setValue(self.parent.settings.value('data')['line_tresh'])
        self.spinHoughLen = QSpinBox(self)
        self.spinHoughLen.setRange(3, 10000)
        self.spinHoughLen.setValue(self.parent.settings.value('data')['line_length'])
        self.spinHoughGap = QSpinBox(self)
        self.spinHoughGap.setRange(0, 5000)
        self.spinHoughGap.setValue(self.parent.settings.value('data')['line_gap'])
        self.spinRasterRes = QSpinBox(self)
        self.spinRasterRes.setRange(10, 10000)
        self.spinRadius = QSpinBox(self)
        self.spinRadius.setRange(10, 100000)
        rasterizLabel = QLabel(
            self.parent.language_dict['commands']['mink_dens_label'][self.parent.selected_language])
        radiusLabel = QLabel(
            self.parent.language_dict['commands']['radius_label'][self.parent.selected_language])
        self.spinRasterRes.setValue(self.parent.settings.value('data')['raster_res'])
        self.spinRadius.setValue(self.parent.settings.value('data')['raster_radius'])
        self.presetsLineamCombo = QComboBox()
        presetsLineamLabel = QLabel(
            self.parent.language_dict['commands']['line_load_preset_label'][self.parent.selected_language])
        # self.presetsLineamCombo.addItems(['small', 'medium', 'many'])
        apply_preset_btn = QPushButton(
            self.parent.language_dict['commands']['line_apply_preset'][self.parent.selected_language])
        apply_preset_btn.clicked.connect(lambda: self.apply_line_preset(self.presetsLineamCombo.currentText()))
        load_line_preset_label = QLabel(
            self.parent.language_dict['commands']['line_load_file_preset_label'][self.parent.selected_language]
        )
        load_line_preset_btn = QPushButton(
            self.parent.language_dict['commands']['line_load_file_preset_btn'][self.parent.selected_language]
        )
        save_line_preset_btn = QPushButton(
            self.parent.language_dict['commands']['line_save_file_preset_btn'][self.parent.selected_language]
        )

        load_line_preset_btn.clicked.connect(self.line_preset_from_file)
        save_line_preset_btn.clicked.connect(self.line_preset_to_file)

        # faults detecting widgets
        tab_fault_name = self.parent.language_dict['commands']['fault_det_set_tab_name'][self.parent.selected_language]
        houghTransformLabelFaults = QLabel(
            self.parent.language_dict['commands']['line_det_hough_label'][self.parent.selected_language])
        houghTransformTreshLabelFaults = QLabel(
            self.parent.language_dict['commands']['line_det_treshold_label'][self.parent.selected_language])
        houghTransformLenLabelFaults = QLabel(
            self.parent.language_dict['commands']['line_det_len_label'][self.parent.selected_language])
        houghTransformGapLabelFaults = QLabel(
            self.parent.language_dict['commands']['line_det_gap_label'][self.parent.selected_language])
        self.spinHoughTreshFaults = QSpinBox(self)
        self.spinHoughTreshFaults.setRange(5, 100)
        self.spinHoughTreshFaults.setValue(self.parent.settings.value('data')['fault_tresh'])
        self.spinHoughLenFaults = QSpinBox(self)
        self.spinHoughLenFaults.setRange(3, 10000)
        self.spinHoughLenFaults.setValue(self.parent.settings.value('data')['fault_length'])
        self.spinHoughGapFaults = QSpinBox(self)
        self.spinHoughGapFaults.setRange(0, 5000)
        self.spinHoughGapFaults.setValue(self.parent.settings.value('data')['fault_gap'])
        self.presetsFaultsCombo = QComboBox()
        presetsFaultsLabel = QLabel(
            self.parent.language_dict['commands']['line_load_preset_label'][self.parent.selected_language])
        # self.presetsFaultsCombo.addItems(['faults'])
        apply_preset_btn_faults = QPushButton(
            self.parent.language_dict['commands']['line_apply_preset'][self.parent.selected_language])
        apply_preset_btn_faults.clicked.connect(lambda: self.apply_faults_preset(self.presetsFaultsCombo.currentText()))

        load_faults_preset_label = QLabel(
            self.parent.language_dict['commands']['fault_load_file_preset_label'][self.parent.selected_language]
        )
        load_faults_preset_btn = QPushButton(
            self.parent.language_dict['commands']['fault_load_file_preset_btn'][self.parent.selected_language]
        )
        save_faults_preset_btn = QPushButton(
            self.parent.language_dict['commands']['fault_save_file_preset_btn'][self.parent.selected_language]
        )

        load_faults_preset_btn.clicked.connect(self.fault_preset_from_file)
        save_faults_preset_btn.clicked.connect(self.fault_preset_to_file)

        # line density widgets
        # fractal dimension analysis
        tab_fdens_name = self.parent.language_dict['commands']['density_fractal_set_tab_name'][
            self.parent.selected_language]
        densWinSizeLabel = QLabel(
            self.parent.language_dict['commands']['density_win_size_label'][self.parent.selected_language])
        fracWinSizeLabel = QLabel(
            self.parent.language_dict['commands']['fractal_win_size_label'][self.parent.selected_language])
        self.spinDensWinSize = QSpinBox(self)
        self.spinDensWinSize.setRange(100, 10000)
        self.spinDensWinSize.setValue(self.parent.settings.value('data')['density_window_size'])
        self.spinFracWinSize = QSpinBox(self)
        self.spinFracWinSize.setRange(100, 10000)
        self.spinFracWinSize.setValue(self.parent.settings.value('data')['fractal_window_size'])
        
        #widgets for Canny edge detection tab
        # fractal dimension analysis
        tab_canny_name = self.parent.language_dict['commands']['canny_set_tab_name'][
            self.parent.selected_language]
        cannySigmaLabel = QLabel(
            self.parent.language_dict['commands']['canny_sigma_label'][self.parent.selected_language])
        cannyLowThreshLabel = QLabel(
            self.parent.language_dict['commands']['canny_low_thresh_label'][self.parent.selected_language])
        cannyHighThreshLabel = QLabel(
            self.parent.language_dict['commands']['canny_high_thresh_label'][self.parent.selected_language])
        self.spinSigma = QDoubleSpinBox(self)
        self.spinSigma.setRange(0.1, 100)
        self.spinSigma.setValue(self.parent.settings.value('data')['canny_sigma'])
        self.spinLowThresh = QSpinBox(self)
        self.spinLowThresh.setRange(1, 100)
        self.spinLowThresh.setValue(self.parent.settings.value('data')['canny_lowt'])
        self.spinHighThresh = QSpinBox(self)
        self.spinHighThresh.setRange(1, 100)
        self.spinHighThresh.setValue(self.parent.settings.value('data')['canny_hight'])
        self.presetsCannyCombo = QComboBox()

        apply_preset_btn_canny = QPushButton(
            self.parent.language_dict['commands']['line_apply_preset'][self.parent.selected_language])
        apply_preset_btn_canny.clicked.connect(lambda: self.apply_canny_preset(self.presetsCannyCombo.currentText()))

        load_faults_preset_label = QLabel(
            self.parent.language_dict['commands']['fault_load_file_preset_label'][self.parent.selected_language]
        )
        load_canny_preset_btn = QPushButton(
            self.parent.language_dict['commands']['fault_load_file_preset_btn'][self.parent.selected_language]
        )
        save_canny_preset_btn = QPushButton(
            self.parent.language_dict['commands']['fault_save_file_preset_btn'][self.parent.selected_language]
        )

        load_canny_preset_btn.clicked.connect(self.canny_preset_from_file)
        save_canny_preset_btn.clicked.connect(self.canny_preset_to_file)


        #TODO tab creating add new
        # create tab components
        # TODO set current tab dependingly of selected procedure https://stackoverflow.com/questions/45828478/how-to-set-current-tab-of-qtabwidget-by-name
        toolBox = QTabWidget()
        pageLineam = QWidget(toolBox)  # page for line detection settings
        layoutLineam = QGridLayout()  # layout for line detection settings
        pageFaults = QWidget(toolBox)  # page for faults detection settings
        layoutFaults = QGridLayout()  # layout for faults detection settings
        pageDens = QWidget(toolBox)  # page for coverage generation settings
        layoutDens = QGridLayout()  # layout for coverage generation settings
        pageEdges = QWidget(toolBox)  # page for coverage generation settings
        layoutCannyEdges = QGridLayout()  # layout for coverage generation settings


        # add widgets to lineaments
        layoutLineam.addWidget(binaryMethodLabel, 0, 0, 1, 3)
        layoutLineam.addWidget(self.radioNone, 1, 0) 
        layoutLineam.addWidget(self.radioCanny, 1, 1)  # Добавляем компоненты
        layoutLineam.addWidget(self.radioFlows, 1, 2)
        layoutLineam.addWidget(self.sigmaCannyLabel, 2, 0)
        layoutLineam.addWidget(self.sigmaCannySpin, 2, 1)
        layoutLineam.addWidget(houghTransformLabel, 3, 0)
        layoutLineam.addWidget(houghTransformTreshLabel, 4, 0)
        layoutLineam.addWidget(self.spinHoughTresh, 4, 1, 1, 2)
        layoutLineam.addWidget(houghTransformLenLabel, 5, 0)
        layoutLineam.addWidget(self.spinHoughLen, 5, 1, 1, 2)
        layoutLineam.addWidget(houghTransformGapLabel, 6, 0)
        layoutLineam.addWidget(self.spinHoughGap, 6, 1, 1, 2)
        layoutLineam.addWidget(presetsLineamLabel, 7, 0)
        layoutLineam.addWidget(self.presetsLineamCombo, 7, 1)
        layoutLineam.addWidget(apply_preset_btn, 7, 2)
        layoutLineam.addWidget(load_line_preset_label, 8, 0)
        layoutLineam.addWidget(load_line_preset_btn, 8, 1)
        layoutLineam.addWidget(save_line_preset_btn, 8, 2)

        def disable_canny_sigma(obj: QSpinBox, val=False):
            print('def disable_canny_sigma')
            obj.setEnabled(val)

        # set radio buttons
        self.radioCanny.clicked.connect(lambda: disable_canny_sigma(self.sigmaCannySpin, val=True))
        self.radioFlows.clicked.connect(lambda: disable_canny_sigma(self.sigmaCannySpin, val=False))
        self.radioNone.clicked.connect(lambda: disable_canny_sigma(self.sigmaCannySpin, val=False))

        # add widgets to faults tab
        layoutFaults.addWidget(houghTransformLabelFaults, 2, 0)
        layoutFaults.addWidget(houghTransformTreshLabelFaults, 3, 0)
        layoutFaults.addWidget(self.spinHoughTreshFaults, 3, 1, 1, 2)
        layoutFaults.addWidget(houghTransformLenLabelFaults, 4, 0)
        layoutFaults.addWidget(self.spinHoughLenFaults, 4, 1, 1, 2)
        layoutFaults.addWidget(houghTransformGapLabelFaults, 5, 0)
        layoutFaults.addWidget(self.spinHoughGapFaults, 5, 1, 1, 2)
        layoutFaults.addWidget(presetsFaultsLabel, 6, 0)
        layoutFaults.addWidget(self.presetsFaultsCombo, 6, 1)
        layoutFaults.addWidget(apply_preset_btn_faults, 6, 2)
        layoutFaults.addWidget(load_faults_preset_label, 7, 0)
        layoutFaults.addWidget(load_faults_preset_btn, 7, 1)
        layoutFaults.addWidget(save_faults_preset_btn, 7, 2)

        # add widgets to density and fractal dimension tab
        layoutDens.addWidget(densWinSizeLabel, 0, 0)
        layoutDens.addWidget(fracWinSizeLabel, 1, 0)
        layoutDens.addWidget(self.spinDensWinSize, 0, 1)
        layoutDens.addWidget(self.spinFracWinSize, 1, 1)
        layoutDens.addWidget(rasterizLabel, 2, 0)
        layoutDens.addWidget(self.spinRasterRes, 2, 1)
        layoutDens.addWidget(radiusLabel, 3, 0)
        layoutDens.addWidget(self.spinRadius, 3, 1)

        # add widgets to Canny edges tab
        #not done
        layoutCannyEdges.addWidget(cannySigmaLabel, 0, 0)
        layoutCannyEdges.addWidget(cannyLowThreshLabel, 1, 0)
        layoutCannyEdges.addWidget(self.spinSigma, 0, 1)
        layoutCannyEdges.addWidget(self.spinLowThresh, 1, 1)
        layoutCannyEdges.addWidget(cannyHighThreshLabel, 2, 0)
        layoutCannyEdges.addWidget(self.spinHighThresh, 2, 1)
        layoutCannyEdges.addWidget(presetsLineamLabel, 7, 0)
        layoutCannyEdges.addWidget(self.presetsCannyCombo, 7, 1)
        layoutCannyEdges.addWidget(apply_preset_btn_canny, 7, 2)
        layoutCannyEdges.addWidget(load_line_preset_label, 8, 0)
        layoutCannyEdges.addWidget(load_line_preset_btn, 8, 1)
        layoutCannyEdges.addWidget(save_line_preset_btn, 8, 2)


        # add tab layouts to page, add tab to toolBox
        pageLineam.setLayout(layoutLineam)
        pageFaults.setLayout(layoutFaults)
        pageDens.setLayout(layoutDens)
        pageEdges.setLayout(layoutCannyEdges)
        toolBox.addTab(pageLineam, tab_line_name)
        toolBox.addTab(pageFaults, tab_fault_name)
        toolBox.addTab(pageDens, tab_fdens_name)
        toolBox.addTab(pageEdges, tab_canny_name)

        # toolBox.setCurrentIndex(0)
        vbox = QVBoxLayout()
        vbox.addWidget(toolBox)
        hbox = QHBoxLayout()
        hbox.addWidget(btnCancel)
        hbox.addWidget(btnOk)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        dict_faults = {
            'line_tresh': 30,
            'line_length': 50,
            'line_gap': 20
        }
        dict_small = {
            'line_sigma': 30,
            'line_tresh': 100,
            'line_length': 10,
            'line_gap': 2
        }
        dict_medium = {
            'line_sigma': 10,
            'line_tresh': 10,
            'line_length': 10,
            'line_gap': 3
        }
        dict_many = {
            'line_sigma': 5,
            'line_tresh': 5,
            'line_length': 10,
            'line_gap': 1
        }
        dict_canny = {
            'canny_sigma': 10,
            'canny_lowt': 10,
            'canny_hight': 30
        }

        # presets default
        self.line_presets = {'small': dict_small, 'medium': dict_medium, 'many': dict_many}
        self.faults_presets = {'faults def': dict_faults}
        self.canny_presets = {'canny def': dict_canny}

        # load presets from disk to COmbo Boxes
        self.load_presets()  # read preset files from disk into dict
        self.presetsLineamCombo.addItems([*self.line_presets])
        self.presetsFaultsCombo.addItems([*self.faults_presets])
        self.presetsCannyCombo.addItems([*self.canny_presets])
        #TODO add loading Canny presets

        # resize window before showing it to user
        self.resize(550, 350)

    def load_presets(self):
        presets_dirs = [self.parent.preset_lines_folder, self.parent.preset_faults_folder]
        for foldname_full in presets_dirs:
            # check filetype
            for fname in os.listdir(foldname_full):
                if fname.endswith('.lpf'):
                    key = fname.replace('.lpf', '')
                    fname_full = os.path.join(foldname_full, fname);
                    if 'faults' in fname_full:
                        new_dict = self.load_file_to_dict(fname_full)
                        self.faults_presets.update({key: new_dict})
                    if 'lines' in fname_full:
                        new_dict = self.load_file_to_dict(fname_full)
                        self.line_presets.update({key: new_dict})
                    if 'edges' in fname_full:
                        new_dict = self.load_file_to_dict(fname_full)
                        self.canny_presets.update({key: new_dict})

    def setWidgetsVals(self):
        self.sigmaCannySpin.setValue(self.parent.settings.value('data')['line_sigma'])
        self.spinHoughTresh.setValue(self.parent.settings.value('data')['line_tresh'])
        self.spinHoughLen.setValue(self.parent.settings.value('data')['line_length'])
        self.spinHoughGap.setValue(self.parent.settings.value('data')['line_gap'])
        self.spinHoughTreshFaults.setValue(self.parent.settings.value('data')['fault_tresh'])
        self.spinHoughLenFaults.setValue(self.parent.settings.value('data')['fault_length'])
        self.spinHoughGapFaults.setValue(self.parent.settings.value('data')['fault_gap'])
        self.spinDensWinSize.setValue(self.parent.settings.value('data')['density_window_size'])
        self.spinFracWinSize.setValue(self.parent.settings.value('data')['fractal_window_size'])
        self.spinRasterRes.setValue(self.parent.settings.value('data')['raster_res'])
        self.spinRadius.setValue(self.parent.settings.value('data')['raster_radius'])

    def apply_line_preset(self, pres_name):
        set_dict = copy.deepcopy(self.line_presets[pres_name])
        self.spinHoughTresh.setValue(set_dict['line_tresh'])
        self.spinHoughLen.setValue(set_dict['line_length'])
        self.spinHoughGap.setValue(set_dict['line_gap'])

    def apply_faults_preset(self, pres_name):
        set_dict = copy.deepcopy(self.faults_presets[pres_name])
        self.spinHoughTreshFaults.setValue(set_dict['line_tresh'])
        self.spinHoughLenFaults.setValue(set_dict['line_length'])
        self.spinHoughGapFaults.setValue(set_dict['line_gap'])

    def apply_canny_preset(self, pres_name):
        set_dict = copy.deepcopy(self.canny_presets[pres_name])
        self.spinSigma.setValue(set_dict['canny_sigma'])
        self.spinLowThresh.setValue(set_dict['canny_lowt'])
        self.spinHighThresh.setValue(set_dict['canny_hight'])

    def applySettings(self):
        print("try to apply settings")
        current_dict = {
            'line_sigma': self.sigmaCannySpin.value(),
            'line_tresh': self.spinHoughTresh.value(),
            'line_length': self.spinHoughLen.value(),
            'line_gap': self.spinHoughGap.value(),
            'fault_tresh': self.spinHoughTreshFaults.value(),
            'fault_length': self.spinHoughLenFaults.value(),
            'fault_gap': self.spinHoughGapFaults.value(),
            'density_window_size': self.spinDensWinSize.value(),
            'fractal_window_size': self.spinFracWinSize.value(),
            'raster_res': self.spinRasterRes.value(),
            'raster_radius': self.spinRadius.value(),
            'canny_sigma':self.spinSigma.value(),
            'canny_lowt':self.spinLowThresh.value(),
            'canny_hight':self.spinHighThresh.value()
        }
        if self.radioCanny.isChecked():
            current_dict.update({'binarization': 'canny'})
            current_dict.update({'line_sigma': self.sigmaCannySpin.value()})
        elif self.radioFlows.isChecked():
            current_dict.update({'binarization': 'flows'})
        else:
            current_dict.update({'binarization': 'none'})
        self.parent.saveSettings(set_dict=current_dict)
        self.close()
        # TODO сделать ресет настроек из пресетов

    def lines_settings_to_dict(self):
        current_dict = {
            'line_sigma': self.sigmaCannySpin.value(),
            'line_tresh': self.spinHoughTresh.value(),
            'line_length': self.spinHoughLen.value(),
            'line_gap': self.spinHoughGap.value(),
            'line_sigma': self.sigmaCannySpin.value()
        }
        return current_dict

    def faults_settings_to_dict(self):
        current_dict = {
            'line_tresh': self.spinHoughTreshFaults.value(),
            'line_length': self.spinHoughLenFaults.value(),
            'line_gap': self.spinHoughGapFaults.value(),
        }
        return current_dict
    
    def canny_settings_to_dict(self):
        current_dict = {
            'canny_sigma': self.spinSigma.value(),
            'canny_lowt': self.spinLowThresh.value(),
            'canny_hight': self.spinHighThresh.value(),
        }
        return current_dict

    def dict_to_line_settings(self, current_dict):
        try:
            self.spinHoughTresh.setValue(current_dict['line_tresh'])
            self.spinHoughLen.setValue(current_dict['line_length'])
            self.spinHoughGap.setValue(current_dict['line_gap'])
            self.sigmaCannySpin.setValue(current_dict['line_sigma'])
        except:
            self.parent.msg_info(self.language_dict['commands']['load_file_preset_error'][self.selected_language])

    def dict_to_fault_settings(self, current_dict):
        try:
            self.spinHoughTreshFaults.setValue(current_dict['line_tresh'])
            self.spinHoughLenFaults.setValue(current_dict['line_length'])
            self.spinHoughGapFaults.setValue(current_dict['line_gap'])
        except:
            self.parent.msg_info(self.language_dict['commands']['load_file_preset_error'][self.selected_language])

    def dict_to_canny_settings(self, current_dict):
        try:
            self.spinSigma.setValue(current_dict['canny_sigma'])
            self.spinLowThresh.setValue(current_dict['canny_lowt'])
            self.spinHighThresh.setValue(current_dict['canny_hight'])
        except:
            self.parent.msg_info(self.language_dict['commands']['load_file_preset_error'][self.selected_language])


    def get_save_preset_filename(self=None, txt="Save preset as"):
        path = QFileDialog.getSaveFileName(self, (txt), '', ("lefa preset file (*.lpf)"))
        if path[0] != '':
            return path[0]
        else:
            return None

    def get_open_preset_filename(self, path=None, txt="Open preset file"):
        path = QFileDialog.getOpenFileName(self, (txt), '', ("lefa preset (*.lpf)"))
        if path[0] != '':
            return path[0]
        else:
            return None

    def save_dict_to_file(self, save_file_name, dict_save):
        with open(save_file_name, 'w') as f:
            for key in [*dict_save]:
                f.writelines(key + '=' + str(dict_save[key]) + '\n')

    def load_file_to_dict(self, load_file_name):
        with open(load_file_name, 'r') as f:
            lines = f.readlines()
        dictout = {}
        for line in lines:
            try:
                dictout.update({line.split('=')[0]: int(line.split('=')[1])})
            except:
                dictout.update({line.split('=')[0]: line.split('=')[1]})
        return dictout

    # save line set btn click
    def line_preset_to_file(self):
        cur_dict = self.lines_settings_to_dict()
        fn = self.get_save_preset_filename()
        if fn != None:
            self.save_dict_to_file(fn, cur_dict)

    # load line set btn click
    def line_preset_from_file(self):
        fn = self.get_open_preset_filename()
        if fn != None:
            cur_dict = self.load_file_to_dict(fn)
            print(cur_dict)
            self.dict_to_line_settings(cur_dict)

    # save fault set btn click
    def fault_preset_to_file(self):
        cur_dict = self.faults_settings_to_dict()
        fn = self.get_save_preset_filename()
        if fn != None:
            self.save_dict_to_file(fn, cur_dict)

    # load fault btn click
    def fault_preset_from_file(self):
        fn = self.get_open_preset_filename()
        if fn != None:
            cur_dict = self.load_file_to_dict(fn)
            print(cur_dict)
            self.dict_to_fault_settings(cur_dict)
    
    # save canny set btn click
    def canny_preset_to_file(self):
        cur_dict = self.canny_settings_to_dict()
        fn = self.get_save_preset_filename()
        if fn != None:
            self.save_dict_to_file(fn, cur_dict)

    # load canny btn click
    def canny_preset_from_file(self):
        fn = self.get_open_preset_filename()
        if fn != None:
            cur_dict = self.load_file_to_dict(fn)
            print(cur_dict)
            self.dict_to_canny_settings(cur_dict)


# TODO LAYERS WINDOW
class Layers(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 150, 355, 280)
        try:
            self.setWindowTitle(
                self.parent.language_dict['commands']['layers_window_title'][self.parent.selected_language])
        except:
            self.setWindowTitle('Project files:')
        # buttons in layers list
        try:
            self.label = QLabel(
                self.parent.language_dict['commands']['layers_window_label'][self.parent.selected_language])
        except:
            self.label = QLabel('Project files:')

        self.layer_list = QListWidget()
        try:
            self.layer_button = QPushButton(
                self.parent.language_dict['commands']['layers_window_show_stat_tbn'][self.parent.selected_language])
            self.layer_remove = QPushButton(
                self.parent.language_dict['commands']['layers_window_remove_layer_tbn'][self.parent.selected_language])
            self.layer_show = QPushButton(
                self.parent.language_dict['commands']['layers_window_show_layer_tbn'][self.parent.selected_language])
        except:
            self.layer_button = QPushButton('Show stats')
            self.layer_remove = QPushButton('Remove')
            self.layer_show = QPushButton('Show')

        # assign actions to buttons
        self.layer_remove.clicked.connect(self.parent.layers_list_remove)
        self.layer_show.clicked.connect(self.parent.on_change1)
        self.layer_button.clicked.connect(self.parent.show_stat)

        layout = QVBoxLayout()
        layoutH = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.layer_list)
        layoutH.addWidget(self.layer_button)
        layoutH.addWidget(self.layer_show)
        layoutH.addWidget(self.layer_remove)
        layout.addLayout(layoutH)

        self.setLayout(layout)

        self.show()
        self.raise_()  # put it on top


# TODO map browser
class MapBrowser(QWidget):  # map browser
    resized = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.scrollArea = QScrollArea()
        self.scrollArea.resize(200, 200)
        self.layout = QVBoxLayout(self)
        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.coords = []

        # coordinate labels
        self.labelX = QLabel()
        self.labelY = QLabel()
        layoutH = QHBoxLayout()
        layoutH.addWidget(self.labelX)
        layoutH.addWidget(self.labelY)

        # nav toolbar  https://stackoverflow.com/questions/49057890/matplotlib-navigationtoolbar2-callbacks-e-g-release-zoom-syntax-example
        self.nt = NavigationToolbar(self.canvas, self)  # (1-где разместить, 2-что отследить)
        self.layout.addWidget(self.nt, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.layout.addWidget(self.scrollArea)
        # self.layout.addWidget(self.labelX)
        # self.layout.addWidget(self.labelY)
        layoutH.addWidget(self.labelX)
        layoutH.addWidget(self.labelY)
        self.layout.addLayout(layoutH)
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.setLayout(self.layout)

        # canvas click
        self.canvas.mpl_connect('button_press_event', self.onClick)
        # resize event
        self.resized.connect(self.onResize)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(MapBrowser, self).resizeEvent(event)

    def onResize(self):
        widget = self.geometry()
        self.scrollArea.setGeometry(0, 0, widget.width(), widget.height())

    def onClick(self, event):  # click on app window
        global ix, iy
        ix, iy = event.xdata, event.ydata

        try:
            # fill window label with values
            self.labelX.setText('X:' + str(int(ix)))
            self.labelY.setText('Y:' + str(int(iy)))
            # if flow orders raster exists, show order, else - srtm
        except TypeError:
            print('Probably click was outside map window?')

        item_list = [item.text() for item in self.parent.layers.layer_list.selectedItems()]

        # if no items selected select first item
        try:
            if len(self.parent.layer_list.selectedItems()) == 0:
                self.parent.layers.layer_list.setCurrentItem(
                    self.parent.layers.layer_list.item(0))  # set selection to first items
            else:
                if (item_list[0] == 'flow orders'):
                    zvalue = self.parent.flow_orders[int(iy), int(ix)]
                    # print(self.parent.flow_orders[int(iy),int(ix)])
                elif (item_list[0] == 'flow directions'):
                    zvalue = self.parent.flow_directions[int(iy), int(ix)]
                elif (item_list[0] == 'srtm'):
                    zvalue = self.parent.srtm[int(iy), int(ix)]
                    # print(self.parent.srtm[int(iy),int(ix)])
                elif (item_list[0] == 'enchanced srtm'):
                    zvalue = self.parent.inflated_dem[int(iy), int(ix)]
                    print(self.parent.inflated_dem[int(iy), int(ix)])
                elif ('basediff' in item_list[0]):
                    key = (item_list[0].split('-')[1])
                    img = self.parent.base_surfaces_diff_dict[key]
                    zvalue = img[int(iy), int(ix)]
                elif ('base-' in item_list[0]):
                    key = int(item_list[0].split('-')[1])
                    img = self.parent.base_surfaces_dict[key]
                    zvalue = img[int(iy), int(ix)]
                # else:
                #    zvalue=self.parent.points_out[int(iy),int(ix)]
                #    #print(self.parent.points_out[int(iy),int(ix)])
                self.parent.labelZ.setText(str(zvalue))
                self.coords.append((ix, iy))
                if len(self.coords) == 2:
                    return self.parent.coords
        except:
            print('Unknown error, possibly no SRTM was opened')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'pylefa_logo061.png')
    app.setWindowIcon(QIcon(path))
    win = Window()
    win.show()
    sys.exit(app.exec())
