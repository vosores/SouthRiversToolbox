__author__ = 'Victor Osores'
__date__ = 'January 2021'
__copyright__ = '(C) 2021, Victor Osores'

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterString,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFileDestination,
                       QgsProcessingParameterFolderDestination)
from qgis import processing
from processing.tools import vector
import numpy as np
import matplotlib.pyplot as plt
from ..metadata import AlgorithmMetadata

class Graphics(AlgorithmMetadata, QgsProcessingAlgorithm):


    METADATA = AlgorithmMetadata.read(__file__,'Graphics')
	    
    INPUT = 'INPUT'
    TITLE = 'TITLE'
    xlabel = 'xlabel'
    xcoordinate = 'xcoordinate'
    ylabel = 'ylabel'
    ycoordinate = 'ycoordinate'
    zcoordinate = 'zcoordinate'
    size_figx = 'size_figx'
    size_figy = 'size_figy'
    ext = 'ext'
    colors = 'colors'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr('Input layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.TITLE,
                self.tr('Title of figure'),
                defaultValue='Figure 1'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterField(
                self.xcoordinate,
                self.tr('Name of X Coordinate'),
                None,
                self.INPUT, QgsProcessingParameterField.Any))


        self.addParameter(
            QgsProcessingParameterString(
                self.xlabel,
                self.tr('X label'),
                defaultValue='Distance in meters [m]'
            )
        )
        
        self.addParameter(
            QgsProcessingParameterField(
                self.ycoordinate,
                self.tr('Name of Y Coordinate Left'),
                None,
                self.INPUT, QgsProcessingParameterField.Any))

        self.addParameter(
            QgsProcessingParameterField(
                self.zcoordinate,
                self.tr('Name of Y Coordinate Rigth'),
                None,
                self.INPUT, QgsProcessingParameterField.Any))


        self.addParameter(
            QgsProcessingParameterString(
                self.ylabel,
                self.tr('Y label'),
                defaultValue='Width in meters [m]'
            )
        )

        width = QgsProcessingParameterNumber(
                    self.size_figx,
                    self.tr('Width of figure'),
                    QgsProcessingParameterNumber.Double,
                    defaultValue=10.0
                )
        width.setFlags(width.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(width)
        
        
        height=QgsProcessingParameterNumber(
                self.size_figy,
                self.tr('Height of figure'),
                QgsProcessingParameterNumber.Double,
                defaultValue=3.0
                )
        height.setFlags(height.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(height)
        
        colors = QgsProcessingParameterEnum(
                    self.colors,
                    self.tr('Figure color'),
                    ['grey', 'colors', 'colors with line'],
                    defaultValue=1
                )
        colors.setFlags(colors.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(colors)

        
        exten = QgsProcessingParameterEnum(
                    self.ext,
                    self.tr('Figure extension'),
                    ['.eps', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz'],
                    defaultValue=3
                )
        exten.setFlags(exten.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(exten)
        
        
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                self.tr('Output layer')
            )
        )
        
    def processAlgorithm(self, parameters, context, feedback):
        

        network = self.parameterAsSource(
            parameters,
            self.INPUT,
            context
        )

        colors = self.parameterAsEnum(
            parameters,
            self.colors,
            context
        )

        ext = self.parameterAsEnum(
            parameters,
            self.ext,
            context
        )
        
        title = self.parameterAsString(
            parameters,
            self.TITLE,
            context
        )
        xlabel = self.parameterAsString(
            parameters,
            self.xlabel,
            context
        )
        ylabel = self.parameterAsString(
            parameters,
            self.ylabel,
            context
        )
        xcoordinate = self.parameterAsString(
            parameters,
            self.xcoordinate,
            context
        )
        ycoordinate = self.parameterAsString(
            parameters,
            self.ycoordinate,
            context
        )

        zcoordinate = self.parameterAsString(
            parameters,
            self.zcoordinate,
            context
        )

        size_figx = self.parameterAsDouble(
            parameters,
            self.size_figx,
            context
        )
        size_figy = self.parameterAsDouble(
            parameters,
            self.size_figy,
            context
        )

        output = self.parameterAsFileOutput(
            parameters,
            self.OUTPUT,
            context
        )
        
        ext_pos = ('.eps', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba', '.svg', '.svgz')

        values_x = vector.values(network, xcoordinate)
        values_y = vector.values(network, ycoordinate)
        values_z = vector.values(network, zcoordinate)

        x = np.array(values_x[xcoordinate], dtype = np.float32)
        y = np.array(values_y[ycoordinate], dtype = np.float32)
        z = np.array(values_z[zcoordinate], dtype = np.float32)

        fig, axs = plt.subplots(figsize=(size_figx,size_figy))
        axs.grid(True,linestyle='--')
        if colors==0:
            axs.stackplot(x,y, baseline='zero',color='dimgrey')
            axs.stackplot(x,-z, baseline='zero',color='silver')
        elif colors==1:
            axs.stackplot(x,y, baseline='zero',color='blue')
            axs.stackplot(x,-z, baseline='zero',color='orange')
            # axs.plot(x,y,x,-z,color='black')
        else:
            axs.stackplot(x,y, baseline='zero',color='blue')
            axs.stackplot(x,-z, baseline='zero',color='orange')
            axs.plot(x,y,x,-z,color='black')

        axs.set(xlabel=xlabel,ylabel=ylabel,title=title)
        
        plt.savefig(output+ext_pos[ext],bbox_inches='tight')
   
        return {self.OUTPUT: output}