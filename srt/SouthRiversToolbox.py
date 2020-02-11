# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SouthRiversToolboxPlugin
                                 A QGIS plugin
                              -------------------
        copyright            : (C) 2020 by vosores
        email                : victorosores@udec.cl
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import os
import importlib
from qgis.PyQt.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
# from PyQt5.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from qgis.core import ( # pylint: disable=import-error,no-name-in-module
    QgsApplication,
    QgsProcessingProvider,
    QgsProcessingAlgorithm
)

from .resources import *
# from .SouthRiversToolbox_dialog import SouthRiversToolboxPluginDialog
import os.path

from processing.core.ProcessingConfig import ProcessingConfig, Setting

from .functions.metadata import AlgorithmMetadata
class SouthRiversToolboxPlugin:
    def __init__(self, iface):
        # self.iface = iface
        self.providers = [cls() for cls in PROVIDERS]

    def initGui(self):
        # from qgis.core import QgsApplication
        for provider in self.providers:
            QgsApplication.processingRegistry().addProvider(provider)

    def unload(self):

        for provider in self.providers:
            QgsApplication.processingRegistry().removeProvider(provider)

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('SouthRiversToolbox', message)

class SouthRiversBaseProvider(QgsProcessingProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = {g['id']: g['displayName'] for g in self.METADATA['groups']}

    def groupDisplayName(self, group):

        return self.groups[group]

    def icon(self):
        return QIcon(os.path.join(os.path.dirname(__file__), self.ICON))

    def loadAlgorithms(self):

        alg_dir = os.path.join(os.path.dirname(__file__), self.SOURCE_FOLDER)

        groups = sorted([
            g for g in os.listdir(alg_dir)
            if not g.startswith('__') and os.path.isdir(os.path.join(alg_dir, g))
        ])

        for group in groups:

            module = importlib.reload(importlib.import_module(f'..{self.SOURCE_FOLDER}.{group}', __name__))
            count = 0

            for key in dir(module):

                obj = getattr(module, key)

                if callable(obj):
                    algorithm = obj()
                    if isinstance(algorithm, QgsProcessingAlgorithm):
                        self.addAlgorithm(algorithm)
                        count += 1

    # # def __init__(self, iface):
    # #     self.iface = iface
    # #     # initialize plugin directory
    # #     self.plugin_dir = os.path.dirname(__file__)
    # #     # initialize locale
    # #     locale = QSettings().value('locale/userLocale')[0:2]
    # #     locale_path = os.path.join(
    # #         self.plugin_dir,
    # #         'i18n',
    # #         'SouthRiversToolboxPlugin_{}.qm'.format(locale))

    # #     if os.path.exists(locale_path):
    # #         self.translator = QTranslator()
    # #         self.translator.load(locale_path)

    # #         if qVersion() > '4.3.3':
    # #             QCoreApplication.installTranslator(self.translator)

    # #     # Create the dialog (after translation) and keep reference
    # #     self.dlg = SouthRiversToolboxPluginDialog()

    # #     # Declare instance attributes
    # #     self.actions = []
    # #     self.menu = self.tr(u'&South Rivers Toolbox')
    # #     # TODO: We are going to let the user set this up in a future iteration
    # #     self.toolbar = self.iface.addToolBar(u'SouthRiversToolboxPlugin')
    # #     self.toolbar.setObjectName(u'SouthRiversToolboxPlugin')

    # # def tr(self, message):
    # #     return QCoreApplication.translate('SouthRiversToolboxPlugin', message)

    # # def add_action(
    # #     self,
    # #     icon_path,
    # #     text,
    # #     callback,
    # #     enabled_flag=True,
    # #     add_to_menu=True,
    # #     add_to_toolbar=True,
    # #     status_tip=None,
    # #     whats_this=None,
    # #     parent=None):
    # #     """Add a toolbar icon to the toolbar.

    # #     :param icon_path: Path to the icon for this action. Can be a resource
    # #         path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
    # #     :type icon_path: str

    # #     :param text: Text that should be shown in menu items for this action.
    # #     :type text: str

    # #     :param callback: Function to be called when the action is triggered.
    # #     :type callback: function

    # #     :param enabled_flag: A flag indicating if the action should be enabled
    # #         by default. Defaults to True.
    # #     :type enabled_flag: bool

    # #     :param add_to_menu: Flag indicating whether the action should also
    # #         be added to the menu. Defaults to True.
    # #     :type add_to_menu: bool

    # #     :param add_to_toolbar: Flag indicating whether the action should also
    # #         be added to the toolbar. Defaults to True.
    # #     :type add_to_toolbar: bool

    # #     :param status_tip: Optional text to show in a popup when mouse pointer
    # #         hovers over the action.
    # #     :type status_tip: str

    # #     :param parent: Parent widget for the new action. Defaults None.
    # #     :type parent: QWidget

    # #     :param whats_this: Optional text to show in the status bar when the
    # #         mouse pointer hovers over the action.

    # #     :returns: The action that was created. Note that the action is also
    # #         added to self.actions list.
    # #     :rtype: QAction
    # #     """

    # #     icon = QIcon(icon_path)
    # #     action = QAction(icon, text, parent)
    # #     action.triggered.connect(callback)
    # #     action.setEnabled(enabled_flag)

    # #     if status_tip is not None:
    # #         action.setStatusTip(status_tip)

    # #     if whats_this is not None:
    # #         action.setWhatsThis(whats_this)

    # #     if add_to_toolbar:
    # #         self.toolbar.addAction(action)

    # #     if add_to_menu:
    # #         # Customize the iface method if you want to add to a specific
    # #         # menu (for example iface.addToVectorMenu):
    # #         self.iface.addPluginToMenu(
    # #             self.menu,
    # #             action)

    # #     self.actions.append(action)

    # #     return action

    # # def initGui(self):
    # #     """Create the menu entries and toolbar icons inside the QGIS GUI."""

    # #     icon_path = ':/plugins/SouthRiversToolbox/images/icon.png'
    # #     self.add_action(
    # #         icon_path,
    # #         text=self.tr(u'South Rivers Toolbox'),
    # #         callback=self.run,
    # #         parent=self.iface.mainWindow())


    # # def unload(self):
    # #     """Removes the plugin menu item and icon from QGIS GUI."""
    # #     for action in self.actions:
    # #         self.iface.removePluginMenu(
    # #             self.tr(u'&South Rivers Toolbox'),
    # #             action)
    # #         self.iface.removeToolBarIcon(action)
    # #     # remove the toolbar
    # #     del self.toolbar


    # # def run(self):
    # #     """Run method that performs all the real work"""
    # #     # show the dialog
    # #     self.dlg.show()
    # #     # Run the dialog event loop
    # #     result = self.dlg.exec_()
    # #     # See if OK was pressed
        
    # #     if result:
    # #         # Do something useful here - delete the line containing pass and
    # #         # substitute with your code.
    # #         pass


# class SouthRiversWorkflowsProvider(SouthRivBaseProvider):

#     # METADATA = AlgorithmMetadata.read(__file__, 'FluvialCorridorWorkflows')
#     SOURCE_FOLDER = 'workflows'
#     ICON = 'images/icon.png'

#     def id(self):
#         return 'srt'

#     def name(self):
#         return 'South Rivers Workflows'
    
#     def longName(self):
#         return 'South Rivers Workflows'

#     def load(self):
#         self.refreshAlgorithms()
#         return True

#     def groupDisplayName(self, group):

#         return self.groups[group]
class SouthRiversToolboxProvider(SouthRiversBaseProvider):

    METADATA = AlgorithmMetadata.read(__file__, 'SouthRiversToolbox')
    SOURCE_FOLDER = 'functions'
    ICON = 'images/icon.png'
    CYTHON_SETTING = 'SRT_ACTIVATE_CYTHON'

    def id(self):
        return 'srt'

    def name(self):
        return 'South Rivers Toolbox'
    
    def longName(self):
        return 'South Rivers Toolbox'

    def load(self):
        
        ProcessingConfig.addSetting(
            Setting(
                self.name(),
                self.CYTHON_SETTING,
                self.tr('Activate Cython Extensions'),
                True))

        ProcessingConfig.readSettings()
        self.refreshAlgorithms()

        return True

    def unload(self):
        ProcessingConfig.removeSetting(self.CYTHON_SETTING)

# # class SouthRiversWorkflowsProvider(SouthRiversBaseProvider):
# #     SOURCE_FOLDER = 'workflows'
# #     ICON = 'images/icon.png'

# #     def id(self):
# #         return 'srt'

# #     def name(self):
# #         return 'South Rivers Workflows'
    
# #     def longName(self):
# #         return 'South Rivers Workflows'

# #     # def load(self):
# #     #     self.refreshAlgorithms()
# #     #     return True

# #     def groupDisplayName(self, group):

# #         return self.groups[group]

PROVIDERS = [
    SouthRiversToolboxProvider#,
    # SouthRiversWorkflowsProvider
]