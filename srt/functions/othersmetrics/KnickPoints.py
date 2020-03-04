# -*- coding: utf-8 -*-

"""
Knick Points Detection

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import math

from qgis.PyQt.QtCore import ( # pylint:disable=no-name-in-module
    QVariant
)

from qgis.core import ( # pylint:disable=no-name-in-module
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsWkbTypes
)

from ..metadata import AlgorithmMetadata

class KnickPoints(AlgorithmMetadata, QgsProcessingAlgorithm):
    """
    Knickpoints detection based on Relative Slope Extension Index (RSE)

    References:

    [1] Seeber, L., & Gornitz, V. (1983).
        River profiles along the Himalayan arc as indicators of active tectonics.
        Tectonophysics, 92(4), 335‑367.
        https://doi.org/10.1016/0040-1951(83)90201-9

    [2] Queiroz et al. (2015).
        Knickpoint finder: A software tool that improves neotectonic analysis.
        Computers & Geosciences, 76, 80‑87.
        https://doi.org/10.1016/j.cageo.2014.11.004

    [3] Knickpoint Finder, ArcGIS implementation
        http://www.neotectonica.ufpr.br/2013/index.php/aplicativos/doc_download/87-knickpointfinder
        No License
    """

    METADATA = AlgorithmMetadata.read(__file__, 'KnickPoints')

    INPUT = 'INPUT'
    NODATA = 'NODATA'
    MIN_DZ = 'MIN_DZ'
    MIN_RSE = 'MIN_RSE'
    MIN_RSE_TOTAL = 'MIN_RSE_TOTAL'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, configuration): #pylint: disable=unused-argument,missing-docstring

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT,
            self.tr('Stream Network Aggregated by Hack Order with Z Coordinate'),
            [QgsProcessing.TypeVectorLine]))

        self.addParameter(QgsProcessingParameterNumber(
            self.NODATA,
            self.tr('No Data Value for Z')))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_DZ,
            self.tr('Contour Interval'),
            minValue=0.0,
            defaultValue=5.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_RSE,
            self.tr('Minimum Knickpoints RSE Value'),
            minValue=0.0,
            defaultValue=2.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_RSE_TOTAL,
            self.tr('Minimum RSE Total Value'),
            minValue=0.0,
            defaultValue=1.0))

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Knickpoints'),
            QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback): #pylint: disable=unused-argument,missing-docstring

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA, context)
        min_dz = self.parameterAsDouble(parameters, self.MIN_DZ, context)
        knickpoint_min_rse = self.parameterAsDouble(parameters, self.MIN_RSE, context)
        min_rse_total = self.parameterAsDouble(parameters, self.MIN_RSE_TOTAL, context)

        if not QgsWkbTypes.hasZ(layer.wkbType()):
            raise QgsProcessingException('Input features must have Z coordinate')

        fields = QgsFields(layer.fields())
        fields.append(QgsField('L', QVariant.Double))
        fields.append(QgsField('H', QVariant.Double))
        fields.append(QgsField('DL', QVariant.Double))
        fields.append(QgsField('DH', QVariant.Double))
        fields.append(QgsField('HGI', QVariant.Double))
        fields.append(QgsField('RSE', QVariant.Double))
        fields.append(QgsField('RSET', QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context,
            fields,
            QgsWkbTypes.PointZ,
            layer.sourceCrs())

        total = 100.0 / layer.featureCount() if layer.featureCount() else 0

        for current, feature in enumerate(layer.getFeatures()):

            if feedback.isCanceled():
                break

            geometry = feature.geometry()
            vertices = [v for v in geometry.vertices()]
            z0 = vertices[0].z()
            profile_height = z0 - vertices[-1].z()
            rse_total = profile_height / max(0.0001, math.log(geometry.length()))

            if rse_total < min_rse_total:
                continue

            stretch_length = 0.0
            upstream_length = 0.0
            previous = vertices[0]

            for vertex in vertices[1:-1]:

                if feedback.isCanceled():
                    break

                if vertex.z() == nodata:
                    continue

                dz = previous.z() - vertex.z()
                if dz < min_dz:
                    continue

                stretch_length += vertex.distance(previous)
                upstream_length += vertex.distance(previous)

                if stretch_length:

                    gradient_index = dz/stretch_length * upstream_length
                    rse_index = gradient_index / max(0.0001, rse_total)

                    if knickpoint_min_rse <= 0 or rse_index >= knickpoint_min_rse:

                        knickpoint = QgsFeature()
                        knickpoint.setGeometry(QgsGeometry(vertex))
                        knickpoint.setAttributes(feature.attributes() + [
                            upstream_length,
                            z0 - vertex.z(),
                            stretch_length,
                            dz,
                            gradient_index,
                            rse_index,
                            rse_total
                        ])

                        sink.addFeature(knickpoint)

                    previous = vertex
                    stretch_length = 0.0

            feedback.setProgress(int(current*total))

        return {
            self.OUTPUT: dest_id
        }
