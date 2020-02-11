# -*- coding: utf-8 -*-
"""
/***************************************************************************
 SouthRiversToolboxPlugin
                                 A QGIS plugin
 Description
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
 This script initializes the plugin, making it known to QGIS.
"""

def classFactory(iface):
    from .SouthRiversToolbox import SouthRiversToolboxPlugin
    return SouthRiversToolboxPlugin(iface)
