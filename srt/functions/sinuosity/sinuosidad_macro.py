# -*- coding: utf-8 -*-

"""
sinuosidad_macro

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os

from qgis.core import ( # pylint:disable=no-name-in-module
    QgsProcessingModelAlgorithm
)

from ..metadata import AlgorithmMetadata

class sinuosidad_macro(AlgorithmMetadata, QgsProcessingModelAlgorithm):
    """ 
    Sinuosidad macro
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.METADATA = AlgorithmMetadata.read(__file__, type(self).__name__)
        self.fromFile(os.path.join(os.path.dirname(__file__), type(self).__name__ + '.model3'))
