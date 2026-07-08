def test_import_freq_and_gls():
    import freq
    from freq.gls import Gls
    assert callable(Gls)
