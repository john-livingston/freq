def test_l1p_imports_quietly(capsys):
    from freq.l1p import l1periodogram_v1, covariance_matrices
    assert hasattr(l1periodogram_v1, 'l1p_class')
    assert hasattr(covariance_matrices, 'covar_mat')
    assert 'gglasso' not in capsys.readouterr().out
