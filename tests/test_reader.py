import pydicom.data

from napari_nibabel import napari_get_reader


def test_dicom(tmp_path):
    # write some fake data using your supported file format
    my_test_file = pydicom.data.get_testdata_file(
        "MR_small.dcm", download=True
    )

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
