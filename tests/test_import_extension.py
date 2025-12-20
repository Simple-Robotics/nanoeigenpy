import nanoeigenpy


def test_import_nanoeigenpy():
    assert hasattr(nanoeigenpy, "EigenSolver")
