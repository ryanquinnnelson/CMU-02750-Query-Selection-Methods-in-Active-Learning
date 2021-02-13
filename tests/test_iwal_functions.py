from iwal.iwal_functions import checkup


def test_checkup():
    actual = checkup()
    expected = 4
    assert(actual, expected)
