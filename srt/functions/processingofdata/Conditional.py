# -*- coding: utf-8 -*-

"""
Conditional

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
from osgeo import gdal

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer
)

from ..metadata import AlgorithmMetadata

class Conditional(AlgorithmMetadata, QgsProcessingAlgorithm):

    METADATA = AlgorithmMetadata.read(__file__, 'Conditional')

    MINLEVEL = 'MINLEVEL'
    MAXLEVEL = 'MAXLEVEL'
    ELEVATIONS = 'ELEVATIONS'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, configuration):

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.ELEVATIONS,
            self.tr('Flow Accumulation')))

        self.addParameter(QgsProcessingParameterNumber(self.MINLEVEL,
                                                       self.tr('Minimum level'),
                                                       type=QgsProcessingParameterNumber.Double,
                                                       minValue=-10000000.0, maxValue=1000000.0, defaultValue=-10000.0, 
                                                       optional=False))

        self.addParameter(QgsProcessingParameterNumber(self.MAXLEVEL,
                                                       self.tr('Maximum level'),
                                                       type=QgsProcessingParameterNumber.Double,
                                                       minValue=-10000000.0, maxValue=1000000.0, defaultValue=10000.0, 
                                                       optional=False))

        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT,
            self.tr('Channels')))

    # def canExecute(self): 

    #     try:
    #         from ...lib.raster import Grid
    #         return True, ''
    #     except ImportError:
    #         return False, self.tr('Missing dependency for Raster')

    def processAlgorithm(self, parameters, context, feedback): 
        # from ...lib.raster import Grid
        minlevel = self.parameterAsDouble(parameters, self.MINLEVEL, context)
        maxlevel = self.parameterAsDouble(parameters, self.MAXLEVEL, context)
        # path = self.parameterAsString(parameters, self.ELEVATIONS, context)
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        driver = gdal.GetDriverByName('GTiff')
        elevations_lyr = self.parameterAsRasterLayer(parameters, self.ELEVATIONS, context)
        elevations_ds = gdal.Open(elevations_lyr.dataProvider().dataSourceUri())
        elevations = elevations_ds.GetRasterBand(1).ReadAsArray()
        nodata = elevations_ds.GetRasterBand(1).GetNoDataValue()
        
        # grid = Grid.open_raster(path, data_name='accu')
        # accu = grid.view('accu')
        for i in range(elevations_ds.RasterXSize):
            for j in range(elevations_ds.RasterYSize):
                if elevations[j,i]<=minlevel:
                    elevations[j,i]=1
                elif elevations[j,i]>=maxlevel:
                    elevations[j,i]=1
                else:
                    elevations[j,i]=0

        accu = elevations.astype(int)

        if feedback.isCanceled():
            feedback.reportError(self.tr('Aborted'), True)
            return {}

        feedback.setProgress(50)
        feedback.pushInfo(self.tr('Write output ...'))

        dst = driver.Create(
            output,
            xsize=elevations_ds.RasterXSize,
            ysize=elevations_ds.RasterYSize,
            bands=1,
            eType=gdal.GDT_Int16,
            options=['TILED=YES', 'COMPRESS=DEFLATE'])

        dst.SetGeoTransform(elevations_ds.GetGeoTransform())
        dst.SetProjection(elevations_lyr.crs().toWkt())
        dst.GetRasterBand(1).WriteArray(accu)
        dst.GetRasterBand(1).SetNoDataValue(-1)

        feedback.setProgress(100)

        # Properly close GDAL resources
        elevations_ds = None
        dst = None

        return {self.OUTPUT: output}
