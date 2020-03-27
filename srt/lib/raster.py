from affine import Affine
import pyproj
import numpy as np
try:
    import rasterio
    import rasterio.features
    _HAS_RASTERIO = True
except:
    _HAS_RASTERIO = False
from scipy import spatial
from scipy import interpolate

class Grid(object):
    """
    Container class for holding and manipulating gridded data.
 
    Attributes
    ==========
    affine : Affine transformation matrix (uses affine module)
    shape : The shape of the grid (number of rows, number of columns).
    bbox : The geographical bounding box of the current view of the gridded data
           (xmin, ymin, xmax, ymax).
    mask : A boolean array used to mask certain grid cells in the bbox;
           may be used to indicate which cells lie inside a catchment.
 
    Methods
    =======
        --------
        File I/O
        --------
        add_gridded_data : Add a gridded dataset (dem, flowdir, accumulation)
                           to Grid instance (generic method).
        read_raster : Read a raster file and add the data to a Grid
                      instance.
        open_raster : Initializes Grid from a raster file.
        to_ascii : Writes current "view" of gridded dataset(s) to ascii file.
        ---------------
        Data Processing
        ---------------
        view : Returns a "view" of a dataset defined by an affine transformation
               self.affine (can optionally be masked with self.mask).
        set_bbox : Sets the bbox of the current "view" (self.bbox).
        set_nodata : Sets the nodata value for a given dataset.
        grid_indices : Returns arrays containing the geographic coordinates
                       of the grid's rows and columns for the current "view".
        nearest_cell : Returns the index (column, row) of the cell closest
                       to a given geographical coordinate (x, y).
    """

    def __init__(self, affine=Affine(0,0,0,0,0,0), shape=(1,1), nodata=0,
                 crs=pyproj.Proj('+init=epsg:4326'),
                 mask=None):
        self.affine = affine
        self.shape = shape
        self.nodata = nodata
        self.crs = crs
        # TODO: Mask should be a raster, not an array
        if mask is None:
            self.mask = np.ones(shape)
        self.grids = []


    def add_gridded_data(self, data, data_name, affine=None, shape=None, crs=None,
                         nodata=None, mask=None, metadata={}):
        """
        A generic method for adding data into a Grid instance.
        Inserts data into a named attribute of Grid (name of attribute
        determined by keyword 'data_name').

        Parameters
        ----------
        data : numpy ndarray
               Data to be inserted into Grid instance.
        data_name : str
                    Name of dataset. Will determine the name of the attribute
                    representing the gridded data.
        affine : affine.Affine
                 Affine transformation matrix defining the cell size and bounding
                 box (see the affine module for more information).
        shape : tuple of int (length 2)
                Shape (rows, columns) of data.
        crs : dict
              Coordinate reference system of gridded data.
        nodata : int or float
                 Value indicating no data in the input array.
        mask : numpy ndarray
               Boolean array indicating which cells should be masked.
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (64, 128, 1, 2, 4, 8, 16, 32),
                             'routing' : 'd8'}
        """
        if isinstance(data, Raster):
            if affine is None:
                affine = data.affine
                shape = data.shape
                crs = data.crs
                nodata = data.nodata
                mask = data.mask
        else:
            if mask is None:
                mask = np.ones(shape, dtype=np.bool)
            if shape is None:
                shape = data.shape
        if not isinstance(data, np.ndarray):
            raise TypeError('Input data must be ndarray')
        # if there are no datasets, initialize bbox, shape,
        # cellsize and crs based on incoming data
        if len(self.grids) < 1:
            # check validity of shape
            if ((hasattr(shape, "__len__")) and (not isinstance(shape, str))
                    and (len(shape) == 2) and (isinstance(sum(shape), int))):
                shape = tuple(shape)
            else:
                raise TypeError('shape must be a tuple of ints of length 2.')
            if crs is not None:
                if isinstance(crs, pyproj.Proj):
                    pass
                elif isinstance(crs, dict) or isinstance(crs, str):
                    crs = pyproj.Proj(crs)
                else:
                    raise TypeError('Valid crs required')
            if isinstance(affine, Affine):
                pass
            else:
                raise TypeError('affine transformation matrix required')
            # initialize instance metadata
            self.affine = affine
            self.shape = shape
            self.crs = crs
            self.nodata = nodata
            self.mask = mask
        # assign new data to attribute; record nodata value
        viewfinder = RegularViewFinder(affine=affine, shape=shape, mask=mask, nodata=nodata,
                                       crs=crs)
        data = Raster(data, viewfinder, metadata=metadata)
        self.grids.append(data_name)
        setattr(self, data_name, data)

    def read_raster(self, data, data_name, band=1, window=None, window_crs=None,
                    metadata={}, **kwargs):
        """
        Reads data from a raster file into a named attribute of Grid
        (name of attribute determined by keyword 'data_name').
 
        Parameters
        ----------
        data : str
               File name or path.
        data_name : str
                    Name of dataset. Will determine the name of the attribute
                    representing the gridded data.
        band : int
               The band number to read if multiband.
        window : tuple
                 If using windowed reading, specify window (xmin, ymin, xmax, ymax).
        window_crs : pyproj.Proj instance
                     Coordinate reference system of window. If None, assume it's in raster's crs.
        metadata : dict
                   Other attributes describing dataset, such as direction
                   mapping for flow direction files. e.g.:
                   metadata={'dirmap' : (64, 128, 1, 2, 4, 8, 16, 32),
                             'routing' : 'd8'}
 
        Additional keyword arguments are passed to rasterio.open()
        """
        # read raster file
        if not _HAS_RASTERIO:
            raise ImportError('Requires rasterio module')
        with rasterio.open(data, **kwargs) as f:
            crs = pyproj.Proj(f.crs, preserve_units=True)
            if window is None:
                shape = f.shape
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band))
                else:
                    data = np.ma.filled(f.read())
                affine = f.transform
                data = data.reshape(shape)
            else:
                if window_crs is not None:
                    if window_crs.srs != crs.srs:
                        xmin, ymin, xmax, ymax = window
                        extent = pyproj.transform(window_crs, crs, (xmin, xmax),
                                                  (ymin, ymax))
                        window = (extent[0][0], extent[1][0], extent[0][1], extent[1][1])
                # If window crs not specified, assume it's in raster crs
                ix_window = f.window(*window)
                if len(f.indexes) > 1:
                    data = np.ma.filled(f.read_band(band, window=ix_window))
                else:
                    data = np.ma.filled(f.read(window=ix_window))
                affine = f.window_transform(ix_window)
                data = np.squeeze(data)
                shape = data.shape
            nodata = f.nodatavals[0]
        if nodata is not None:
            nodata = data.dtype.type(nodata)
        self.add_gridded_data(data=data, data_name=data_name, affine=affine, shape=shape,
                              crs=crs, nodata=nodata, metadata=metadata)

    @classmethod
    def open_raster(cls, path, data_name, **kwargs):
        newinstance = cls()
        newinstance.read_raster(path, data_name, **kwargs)
        return newinstance

    def view(self, data, data_view=None, target_view=None, apply_mask=True,
             nodata=None, interpolation='nearest', as_crs=None, return_coords=False,
             kx=3, ky=3, s=0, tolerance=1e-3, dtype=None, metadata={}):
        """
        Return a copy of a gridded dataset clipped to the current "view". The view is determined by
        an affine transformation which describes the bounding box and cellsize of the grid.
        The view will also optionally mask grid cells according to the boolean array self.mask.
 
        Parameters
        ----------
        data : str or Raster
               If str: name of the dataset to be viewed.
               If Raster: a Raster instance (see pysheds.view.Raster)
        data_view : RegularViewFinder or IrregularViewFinder
                    The view at which the data is defined (based on an affine
                    transformation and shape). Defaults to the Raster dataset's
                    viewfinder attribute.
        target_view : RegularViewFinder or IrregularViewFinder
                      The desired view (based on an affine transformation and shape)
                      Defaults to a viewfinder based on self.affine and self.shape.
        apply_mask : bool
               If True, "mask" the view using self.mask.
        nodata : int or float
                 Value indicating no data in output array.
                 Defaults to the `nodata` attribute of the input dataset.
        interpolation: 'nearest', 'linear', 'cubic', 'spline'
                       Interpolation method to be used. If both the input data
                       view and output data view can be defined on a regular grid,
                       all interpolation methods are available. If one
                       of the datasets cannot be defined on a regular grid, or the
                       datasets use a different CRS, only 'nearest', 'linear' and
                       'cubic' are available.
        as_crs: pyproj.Proj
                Projection at which to view the data (overrides self.crs).
        return_coords: bool
                       If True, return the coordinates corresponding to each value
                       in the output array.
        kx, ky: int
                Degrees of the bivariate spline, if 'spline' interpolation is desired.
        s : float
            Smoothing factor of the bivariate spline, if 'spline' interpolation is desired.
        tolerance: float
                   Maximum tolerance when matching coordinates. Data coordinates
                   that cannot be matched to a target coordinate within this
                   tolerance will be masked with the nodata value in the output array.
        dtype: numpy datatype
               Desired datatype of the output array.
        """
        # Check interpolation method
        try:
            interpolation = interpolation.lower()
            assert(interpolation in ('nearest', 'linear', 'cubic', 'spline'))
        except:
            raise ValueError("Interpolation method must be one of: "
                             "'nearest', 'linear', 'cubic', 'spline'")
        # Parse data
        if isinstance(data, str):
            data = getattr(self, data)
            if nodata is None:
                nodata = data.nodata
            if data_view is None:
                data_view = data.viewfinder
            metadata.update(data.metadata)
        elif isinstance(data, Raster):
            if nodata is None:
                nodata = data.nodata
            if data_view is None:
                data_view = data.viewfinder
            metadata.update(data.metadata)
        else:
            # If not using a named dataset, make sure the data and view are properly defined
            try:
                assert(isinstance(data, np.ndarray))
            except:
                raise
            # TODO: Should convert array to dataset here
            if nodata is None:
                nodata = data_view.nodata
        # If no target view provided, construct one based on grid parameters
        if target_view is None:
            target_view = RegularViewFinder(affine=self.affine, shape=self.shape,
                                            mask=self.mask, crs=self.crs, nodata=nodata)
        # If viewing at a different crs, convert coordinates
        if as_crs is not None:
            assert(isinstance(as_crs, pyproj.Proj))
            target_coords = target_view.coords
            new_x, new_y = pyproj.transform(target_view.crs, as_crs,
                                            target_coords[:,1], target_coords[:,0])
            # TODO: In general, crs conversion will yield irregular grid (though not necessarily)
            target_view = IrregularViewFinder(coords=np.column_stack([new_y, new_x]),
                                            shape=target_view.shape, crs=as_crs,
                                            nodata=target_view.nodata)
        # Specify mask
        mask = target_view.mask
        # Make sure views are ViewFinder instances
        assert(issubclass(type(data_view), BaseViewFinder))
        assert(issubclass(type(target_view), BaseViewFinder))
        same_crs = target_view.crs.srs == data_view.crs.srs
        # If crs does not match, convert coords of data array to target array
        if not same_crs:
            data_coords = data_view.coords
            # TODO: x and y order might be different
            new_x, new_y = pyproj.transform(data_view.crs, target_view.crs,
                                            data_coords[:,1], data_coords[:,0])
            # TODO: In general, crs conversion will yield irregular grid (though not necessarily)
            data_view = IrregularViewFinder(coords=np.column_stack([new_y, new_x]),
                                            shape=data_view.shape, crs=target_view.crs,
                                            nodata=data_view.nodata)
        # Check if data can be described by regular grid
        data_is_grid = isinstance(data_view, RegularViewFinder)
        view_is_grid = isinstance(target_view, RegularViewFinder)
        # If data is on a grid, use the following speedup
        if data_is_grid and view_is_grid:
            # If doing nearest neighbor search, use fast sorted search
            if interpolation == 'nearest':
                array_view = RegularGridViewer._view_affine(data, data_view, target_view)
            # If spline interpolation is needed, use RectBivariate
            elif interpolation == 'spline':
                # If latitude/longitude, use RectSphereBivariate
                if target_view.crs.is_latlong():
                    array_view = RegularGridViewer._view_rectspherebivariate(data, data_view,
                                                                             target_view,
                                                                             x_tolerance=tolerance,
                                                                             y_tolerance=tolerance,
                                                                             kx=kx, ky=ky, s=s)
                # If not latitude/longitude, use RectBivariate
                else:
                    array_view = RegularGridViewer._view_rectbivariate(data, data_view,
                                                                       target_view,
                                                                       x_tolerance=tolerance,
                                                                       y_tolerance=tolerance,
                                                                       kx=kx, ky=ky, s=s)
            # If some other interpolation method is needed, use griddata
            else:
                array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                                method=interpolation)
        # If either view is irregular, use griddata
        else:
            array_view = IrregularGridViewer._view_griddata(data, data_view, target_view,
                                                            method=interpolation)
        # TODO: This could be dangerous if it returns an irregular view
        array_view = Raster(array_view, target_view, metadata=metadata)
        # Ensure masking is safe by checking datatype
        if dtype is None:
            dtype = max(np.min_scalar_type(nodata), data.dtype)
            # For matplotlib imshow compatibility
            if issubclass(dtype.type, np.floating):
                dtype = max(dtype, np.dtype(np.float32))
        array_view = array_view.astype(dtype)
        # Apply mask
        if apply_mask:
            np.place(array_view, ~mask, nodata)
        # Return output
        if return_coords:
            return array_view, target_view.coords
        else:
            return array_view



class Raster(np.ndarray):
    def __new__(cls, input_array, viewfinder, metadata=None):
        obj = np.asarray(input_array).view(cls)
        try:
            assert(issubclass(type(viewfinder), BaseViewFinder))
        except:
            raise ValueError("Must initialize with a ViewFinder")
        obj.viewfinder = viewfinder
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.viewfinder = getattr(obj, 'viewfinder', None)
        self.metadata = getattr(obj, 'metadata', None)

    @property
    def bbox(self):
        return self.viewfinder.bbox
    @property
    def coords(self):
        return self.viewfinder.coords
    @property
    def view_shape(self):
        return self.viewfinder.shape
    @property
    def mask(self):
        return self.viewfinder.mask
    @property
    def nodata(self):
        return self.viewfinder.nodata
    @nodata.setter
    def nodata(self, new_nodata):
        self.viewfinder.nodata = new_nodata
    @property
    def crs(self):
        return self.viewfinder.crs
    @property
    def view_size(self):
        return np.prod(self.viewfinder.shape)
    @property
    def extent(self):
        bbox = self.viewfinder.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent
    @property
    def cellsize(self):
        dy, dx = self.dy_dx
        cellsize = (dy + dx) / 2
        return cellsize
    @property
    def affine(self):
        return self.viewfinder.affine
    @property
    def properties(self):
        property_dict = {
            'affine' : self.viewfinder.affine,
            'bbox' : self.viewfinder.bbox,
            'shape' : self.viewfinder.shape,
            'crs' : self.viewfinder.crs,
            'nodata' : self.viewfinder.nodata
        }
        return property_dict
    @property
    def dy_dx(self):
        return (-self.affine.e, self.affine.a)

class BaseViewFinder():
    def __init__(self, shape=None, mask=None, nodata=None,
                 crs=pyproj.Proj('+init=epsg:4326'), y_coord_ix=0, x_coord_ix=1):
        if shape is not None:
            self.shape = shape
        else:
            self.shape = (0,0)
        self.crs = crs
        if nodata is None:
            self.nodata = np.nan
        else:
            self.nodata = nodata
        if mask is None:
            self.mask = np.ones(shape).astype(bool)
        else:
            self.mask = mask
        self.y_coord_ix = y_coord_ix
        self.x_coord_ix = x_coord_ix

    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape
    @property
    def mask(self):
        return self._mask
    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask
    @property
    def nodata(self):
        return self._nodata
    @nodata.setter
    def nodata(self, new_nodata):
        self._nodata = new_nodata
    @property
    def crs(self):
        return self._crs
    @crs.setter
    def crs(self, new_crs):
        self._crs = new_crs
    @property
    def size(self):
        return np.prod(self.shape)

class RegularViewFinder(BaseViewFinder):
    def __init__(self, affine, shape, mask=None, nodata=None,
                 crs=pyproj.Proj('+init=epsg:4326'),
                 y_coord_ix=0, x_coord_ix=1):
        if affine is not None:
            self.affine = affine
        else:
            self.affine = Affine(0,0,0,0,0,0)
        super().__init__(shape=shape, mask=mask, nodata=nodata, crs=crs,
                         y_coord_ix=y_coord_ix, x_coord_ix=x_coord_ix)

    @property
    def bbox(self):
        shape = self.shape
        xmin, ymax = self.affine * (0,0)
        xmax, ymin = self.affine * (shape[1] + 1, shape[0] + 1)
        _bbox = (xmin, ymin, xmax, ymax)
        return _bbox
    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent
    @property
    def affine(self):
        return self._affine
    @affine.setter
    def affine(self, new_affine):
        assert(isinstance(new_affine, Affine))
        self._affine = new_affine
    @property
    def coords(self):
        coordinates = np.meshgrid(*self.grid_indices(), indexing='ij')
        return np.vstack(np.dstack(coordinates))
    @coords.setter
    def coords(self, new_coords):
        pass
    @property
    def dy_dx(self):
        return (-self.affine.e, self.affine.a)
    @property
    def properties(self):
        property_dict = {
            'shape' : self.shape,
            'crs' : self.crs,
            'nodata' : self.nodata,
            'affine' : self.affine,
            'bbox' : self.bbox
        }
        return property_dict

    def grid_indices(self, affine=None, shape=None, col_ascending=True, row_ascending=False):
        """
        Return row and column coordinates of a bounding box at a
        given cellsize.
 
        Parameters
        ----------
        shape : tuple of ints (length 2)
                The shape of the 2D array (rows, columns). Defaults
                to instance shape.
        precision : int
                    Precision to use when matching geographic coordinates.
        """
        if affine is None:
            affine = self.affine
        if shape is None:
            shape = self.shape
        y_ix = np.arange(shape[0])
        x_ix = np.arange(shape[1])
        if row_ascending:
            y_ix = y_ix[::-1]
        if not col_ascending:
            x_ix = x_ix[::-1]
        x, _ = affine * np.vstack([x_ix, np.zeros(shape[1])])
        _, y = affine * np.vstack([np.zeros(shape[0]), y_ix])
        return y, x

    def move_window(self, dxmin, dymin, dxmax, dymax):
        """
        Move bounding box window by integer indices
        """
        cell_height, cell_width  = self.dy_dx
        nrows_old, ncols_old = self.shape
        xmin_old, ymin_old, xmax_old, ymax_old = self.bbox
        new_bbox = (xmin_old + dxmin*cell_width, ymin_old + dymin*cell_height,
                    xmax_old + dxmax*cell_width, ymax_old + dymax*cell_height)
        new_shape = (nrows_old + dymax - dymin,
                     ncols_old + dxmax - dxmin)
        new_mask = np.ones(new_shape).astype(bool)
        mask_values = self._mask[max(dymin, 0):min(nrows_old + dymax, nrows_old),
                                 max(dxmin, 0):min(ncols_old + dxmax, ncols_old)]
        new_mask[max(0, dymax):max(0, dymax) + mask_values.shape[0],
                 max(0, -dxmin):max(0, -dxmin) + mask_values.shape[1]] = mask_values
        self.bbox = new_bbox
        self.shape = new_shape
        self.mask = new_mask

class IrregularViewFinder(BaseViewFinder):
    def __init__(self, coords, shape=None, mask=None, nodata=None,
                 crs=pyproj.Proj('+init=epsg:4326'),
                 y_coord_ix=0, x_coord_ix=1):
        if coords is not None:
            self.coords = coords
        else:
            self.coords = np.asarray([0, 0]).reshape(1, 2)
        if shape is None:
            shape = len(coords)
        super().__init__(shape=shape, mask=mask, nodata=nodata, crs=crs,
                         y_coord_ix=y_coord_ix, x_coord_ix=x_coord_ix)
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, new_coords):
        self._coords = new_coords
    @property
    def bbox(self):
        ymin = self.coords[:, self.y_coord_ix].min()
        ymax = self.coords[:, self.y_coord_ix].max()
        xmin = self.coords[:, self.x_coord_ix].min()
        xmax = self.coords[:, self.x_coord_ix].max()
        return xmin, ymin, xmax, ymax
    @bbox.setter
    def bbox(self, new_bbox):
        pass
    @property
    def extent(self):
        bbox = self.bbox
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return extent

class RegularGridViewer():
    def __init__(self):
        pass

    @classmethod
    def _view_df(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        nodata = target_view.nodata
        viewrows, viewcols = target_view.grid_indices()
        rows, cols = data_view.grid_indices()
        view = (pd.DataFrame(data, index=rows, columns=cols)
                .reindex(selfrows, tolerance=y_tolerance, method='nearest')
                .reindex(selfcols, axis=1, tolerance=x_tolerance,
                         method='nearest')
                .fillna(nodata).values)
        return view

    @classmethod
    def _view_kd(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        """
        Appropriate if:
            - Grid is regular
            - Data is regular
            - Grid and data have same cellsize OR no interpolation is needed
        """
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewrows, viewcols = target_view.grid_indices()
        rows, cols = data_view.grid_indices()
        ytree = spatial.cKDTree(rows[:, None])
        xtree = spatial.cKDTree(cols[:, None])
        ydist, y_ix = ytree.query(viewrows[:, None])
        xdist, x_ix = xtree.query(viewcols[:, None])
        y_passed = ydist < y_tolerance
        x_passed = xdist < x_tolerance
        view[np.ix_(y_passed, x_passed)] = data[y_ix[y_passed]][:, x_ix[x_passed]]
        return view

    @classmethod
    def _view_affine(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata, dtype=data.dtype)
        viewrows, viewcols = target_view.grid_indices()
        _, target_row_ix = ~data_view.affine * np.vstack([np.zeros(target_view.shape[0]), viewrows])
        target_col_ix, _ = ~data_view.affine * np.vstack([viewcols, np.zeros(target_view.shape[1])])
        y_ix = np.around(target_row_ix).astype(int)
        x_ix = np.around(target_col_ix).astype(int)
        y_passed = ((np.abs(y_ix - target_row_ix) < y_tolerance)
                    & (y_ix < data_view.shape[0]) & (y_ix >= 0))
        x_passed = ((np.abs(x_ix - target_col_ix) < x_tolerance)
                    & (x_ix < data_view.shape[1]) & (x_ix >= 0))
        view[np.ix_(y_passed, x_passed)] = data[y_ix[y_passed]][:, x_ix[x_passed]]
        return view

    @classmethod
    def _view_kd_2d(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        viewrows, viewcols = target_view.grid_indices()
        rows, cols = data_view.grid_indices()
        row_bool = (rows <= t_ymax + y_tolerance) & (rows >= t_ymin - y_tolerance)
        col_bool = (cols <= t_xmax + x_tolerance) & (cols >= t_xmin - x_tolerance)
        yx_tree = np.vstack(np.dstack(np.meshgrid(rows[row_bool], cols[col_bool], indexing='ij')))
        yx_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        tree = spatial.cKDTree(yx_tree)
        yx_dist, yx_ix = tree.query(yx_query)
        yx_passed = yx_dist < yx_tolerance
        view.flat[yx_passed] = data[np.ix_(row_bool, col_bool)].flat[yx_ix[yx_passed]]
        return view

    @classmethod
    def _view_rectbivariate(cls, data, data_view, target_view, kx=3, ky=3, s=0,
                            x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        target_dx, target_dy = target_view.affine.a, target_view.affine.e
        data_dx, data_dy = data_view.affine.a, data_view.affine.e
        viewrows, viewcols = target_view.grid_indices(col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.grid_indices(col_ascending=True,
                                            row_ascending=True)
        viewrows += target_dy
        viewcols += target_dx
        rows += data_dy
        cols += data_dx
        row_bool = (rows <= t_ymax + y_tolerance) & (rows >= t_ymin - y_tolerance)
        col_bool = (cols <= t_xmax + x_tolerance) & (cols >= t_xmin - x_tolerance)
        rbs_interpolator = (interpolate.
                            RectBivariateSpline(rows[row_bool],
                                                cols[col_bool],
                                                data[np.ix_(row_bool[::-1], col_bool)],
                                                kx=kx, ky=ky, s=s))
        xy_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        view = rbs_interpolator.ev(xy_query[:,0], xy_query[:,1]).reshape(target_view.shape)
        return view

    @classmethod
    def _view_rectspherebivariate(cls, data, data_view, target_view, coords_in_radians=False,
                                  kx=3, ky=3, s=0, x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        target_dx, target_dy = target_view.affine.a, target_view.affine.e
        data_dx, data_dy = data_view.affine.a, data_view.affine.e
        viewrows, viewcols = target_view.grid_indices(col_ascending=True,
                                                      row_ascending=True)
        rows, cols = data_view.grid_indices(col_ascending=True,
                                            row_ascending=True)
        viewrows += target_dy
        viewcols += target_dx
        rows += data_dy
        cols += data_dx
        row_bool = (rows <= t_ymax + y_tolerance) & (rows >= t_ymin - y_tolerance)
        col_bool = (cols <= t_xmax + x_tolerance) & (cols >= t_xmin - x_tolerance)
        if not coords_in_radians:
            rows = np.radians(rows) + np.pi/2
            cols = np.radians(cols) + np.pi
            viewrows = np.radians(viewrows) + np.pi/2
            viewcols = np.radians(viewcols) + np.pi
        rsbs_interpolator = (interpolate.
                            RectBivariateSpline(rows[row_bool],
                                                cols[col_bool],
                                                data[np.ix_(row_bool[::-1], col_bool)],
                                                kx=kx, ky=ky, s=s))
        xy_query = np.vstack(np.dstack(np.meshgrid(viewrows, viewcols, indexing='ij')))
        view = rsbs_interpolator.ev(xy_query[:,0], xy_query[:,1]).reshape(target_view.shape)
        return view

class IrregularGridViewer():
    def __init__(self):
        pass

    @classmethod
    def _view_kd_2d(cls, data, data_view, target_view, x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        row_bool = ((datacoords[:,0] <= t_ymax + y_tolerance) &
                    (datacoords[:,0] >= t_ymin - y_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + x_tolerance) &
                    (datacoords[:,1] >= t_xmin - x_tolerance))
        yx_tree = datacoords[row_bool & col_bool]
        tree = spatial.cKDTree(yx_tree)
        yx_dist, yx_ix = tree.query(viewcoords)
        yx_passed = yx_dist <= yx_tolerance
        view.flat[yx_passed] = data.flat[row_bool & col_bool].flat[yx_ix[yx_passed]]
        return view

    @classmethod
    def _view_griddata(cls, data, data_view, target_view, method='nearest',
                       x_tolerance=1e-3, y_tolerance=1e-3):
        t_xmin, t_ymin, t_xmax, t_ymax = target_view.bbox
        d_xmin, d_ymin, d_xmax, d_ymax = data_view.bbox
        nodata = target_view.nodata
        view = np.full(target_view.shape, nodata)
        viewcoords = target_view.coords
        datacoords = data_view.coords
        yx_tolerance = np.sqrt(x_tolerance**2 + y_tolerance**2)
        row_bool = ((datacoords[:,0] <= t_ymax + y_tolerance) &
                    (datacoords[:,0] >= t_ymin - y_tolerance))
        col_bool = ((datacoords[:,1] <= t_xmax + x_tolerance) &
                    (datacoords[:,1] >= t_xmin - x_tolerance))
        yx_grid = datacoords[row_bool & col_bool]
        view = interpolate.griddata(yx_grid,
                                    data.flat[row_bool & col_bool],
                                    viewcoords, method=method,
                                    fill_value=nodata)
        view = view.reshape(target_view.shape)
        return view
