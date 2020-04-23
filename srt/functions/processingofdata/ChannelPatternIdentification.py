# -*- coding: utf-8 -*-

"""
ChannelPatternIdentification

single-thread and multi-thread

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
import processing
# from osgeo import gdal

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsVectorLayer
)

from ..metadata import AlgorithmMetadata

class ChannelPatternIdentification(AlgorithmMetadata, QgsProcessingAlgorithm):

    METADATA = AlgorithmMetadata.read(__file__, 'ChannelPatternIdentification')

    VECTORS = 'VECTORS'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, configuration):

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.VECTORS,
                self.tr('Input vector layer'),
                types=[QgsProcessing.TypeVectorAnyGeometry]
            )
        )

        # self.addParameter(QgsProcessingParameterString(self.NAME_OF_FIELD,
        #                                                 self.tr('Name of the new FIELD'),
        #                                                 optional=False,defaultValue='New_Campo')
        #                                                 )



        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT,
            self.tr('New Table with Added Field')))

# 
    def processAlgorithm(self, parameters, context, feedback): 

        NAME_OF_FIELD = 'T_channel'
        output = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        salda = processing.run('qgis:addfieldtoattributestable',
                                {
                                'INPUT': parameters[self.VECTORS],
                                'FIELD_NAME': NAME_OF_FIELD,
                                'FIELD_TYPE': 2,
                                'FIELD_LENGTH': 13,
                                'FIELD_PRECISION': 0,
                                'OUTPUT': parameters[self.OUTPUT]
                                })

        layer = QgsVectorLayer(salda['OUTPUT'], "ogr")
        features = layer.getFeatures()

        newValue = 0.0
        dic = {}
        i = 0 #counting
        for feature in features:
            dic[feature['fid']] = feature['Rank_DGO']
            i = i+1

        n_values = i
        Rank_DGO = np.zeros(n_values)

        x = np.zeros(n_values)
        for i in range(n_values):
            Rank_DGO[i] = dic[i+1] #float points

        for i in range(n_values):
            x[i] = np.count_nonzero(Rank_DGO == Rank_DGO[i])   

        layer = QgsVectorLayer(salda['OUTPUT'], "ogr")
        features = layer.getFeatures()

        i=0
        for field in layer.fields():
            if (field.name()==NAME_OF_FIELD):
                break
            i=i+1
        pos = i

        layer.startEditing()
        # with edit(layer):
        i = 0
        for feature in features:
            newValue = int(x[i])
            # feature[NAME_OF_FIELD] = newValue
            if (newValue > 1):
                layer.changeAttributeValue(feature.id(),pos,'multi-thread')
            else:
                layer.changeAttributeValue(feature.id(),pos,'single-thread')
            i = i+1
            # layer.updateFeature(feature)
        layer.commitChanges()


        if feedback.isCanceled():
            feedback.reportError(self.tr('Aborted'), True)
            return {}

        feedback.setProgress(50)
        feedback.pushInfo('write')
 

        return {'OUTPUT': salda['OUTPUT']}
