"""
This reader exposes NiBabel and pydicom readers to napari.
"""
from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from typing import TypeAlias

import nibabel as nib
import numpy as np
import pydicom

IGNORE = (".DS_Store",)

EXTENSIONS = (
    ".par",  # Philips PAR/REC files
    ".hdr",  # hdr/img files (ANALYZE or NIfTI)
    ".nii",  # NIfTI
    ".nii.gz",  # NIfTI (compressed)
    ".gii",  # GIfTI
    ".dcm",  # DICOM
)
StrOrList: TypeAlias = Union[str, List[str]]
LayerData: TypeAlias = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
LayerDataList: TypeAlias = List[
    Union[
        Tuple[LayerData,],
        Tuple[LayerData, Optional[Dict[Any, Any]], str],
    ]
]


def napari_get_reader(
    path: StrOrList,
) -> Optional[Callable[[StrOrList], LayerDataList]]:
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    paths = _flat_paths(paths)
    if any(
        os.path.splitext(p)[1] not in EXTENSIONS
        for p in paths
        if os.path.split(p)[1] not in IGNORE
    ):
        return None
    # return the *function* that can read ``path``.
    return reader_function


def reader_function(path: StrOrList) -> LayerDataList:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    paths = _flat_paths(paths)

    to_try_pydicom = []
    layers: LayerDataList = []
    # first try to load paths with nibabel
    for p in paths:
        try:
            ds = nib.load(p)
        except nib.filebasedimages.ImageFileError:
            to_try_pydicom.append(p)
        else:
            if isinstance(ds, nib.GiftiImage):
                # TODO: add support for colors
                # from what I see GIFTI specifies actual colors for the nodes
                # where napari expects scalar values and a colormap
                points = ds.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
                triangles = ds.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
                assert (
                    len(points) == 1 and len(triangles) == 1
                ), "unsupported GIFTI dataset"
                layers.append(
                    ((points[0].data, triangles[0].data), {}, "surface")
                )
            else:
                data: np.ndarray = ds.get_fdata()
                layers.append((data,))

    # any images that fail to load with nibabel, try loading with pydicom
    dicom_datasets: Dict[str, List[pydicom.Dataset]] = {}
    for p in to_try_pydicom:
        try:
            ds = pydicom.dcmread(p)
            dicom_datasets.setdefault(ds.SeriesInstanceUID, []).append(ds)
        except Exception:
            raise ValueError(f"unable to read data in file '{p}'")

    for uid, ds_list in dicom_datasets.items():
        # return image layers that are stacks based on SeriesInstanceUID
        if len(ds_list) > 1:
            ds_list.sort(key=lambda x: list(x.ImagePositionPatient))
            layers.append(
                (np.stack(tuple(ds.pixel_array for ds in ds_list), 0),),
            )
        else:
            layers.append((ds.pixel_array,))

    return layers


def _flat_paths(paths: List[str]) -> List[str]:
    flat_paths: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            flat_paths.extend(
                os.path.join(p, f) for f in os.listdir(p) if f not in IGNORE
            )
        else:
            flat_paths.append(p)
    return flat_paths
