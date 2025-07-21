import pytest
from truco_setup.rules import (
    max_points,
    envido_points_accept,
    envido_points_reject,
    valid_envido_order,
    get_envido_points_on_accept,
    get_envido_points_on_reject,
    truco_points_accept,
    truco_points_reject,
    valid_truco_order,
    get_truco_points_on_accept,
    get_truco_points_on_reject
)


class TestConstants:
    def test_max_points(self):
        assert max_points == 30

    def test_envido_points_accept_dict(self):
        assert envido_points_accept == {
            "Envido": 2,
            "RealEnvido": 5,
            "FaltaEnvido": "game"
        }

    def test_envido_points_reject_dict(self):
        assert envido_points_reject == {
            "Envido": 1,
            "RealEnvido": 3,
            "FaltaEnvido": 5
        }

    def test_valid_envido_order(self):
        assert valid_envido_order == ["Envido", "RealEnvido", "FaltaEnvido"]

    def test_truco_points_accept_dict(self):
        assert truco_points_accept == {
            "Truco": 2,
            "ReTruco": 3,
            "ValeCuatro": 4
        }

    def test_truco_points_reject_dict(self):
        assert truco_points_reject == {
            "Truco": 1,
            "ReTruco": 2,
            "ValeCuatro": 3
        }

    def test_valid_truco_order(self):
        assert valid_truco_order == ["Truco", "ReTruco", "ValeCuatro"]


class TestEnvidoFunctions:
    def test_get_envido_points_on_accept_valid(self):
        assert get_envido_points_on_accept("Envido") == 2
        assert get_envido_points_on_accept("RealEnvido") == 5
        assert get_envido_points_on_accept("FaltaEnvido") == "game"

    def test_get_envido_points_on_accept_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized envido stage: InvalidEnvido"):
            get_envido_points_on_accept("InvalidEnvido")

    def test_get_envido_points_on_reject_valid(self):
        assert get_envido_points_on_reject("Envido") == 1
        assert get_envido_points_on_reject("RealEnvido") == 3
        assert get_envido_points_on_reject("FaltaEnvido") == 5

    def test_get_envido_points_on_reject_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized envido stage: BadEnvido"):
            get_envido_points_on_reject("BadEnvido")


class TestTrucoFunctions:
    def test_get_truco_points_on_accept_valid(self):
        assert get_truco_points_on_accept("Truco") == 2
        assert get_truco_points_on_accept("ReTruco") == 3
        assert get_truco_points_on_accept("ValeCuatro") == 4

    def test_get_truco_points_on_accept_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized truco stage: InvalidTruco"):
            get_truco_points_on_accept("InvalidTruco")

    def test_get_truco_points_on_reject_valid(self):
        assert get_truco_points_on_reject("Truco") == 1
        assert get_truco_points_on_reject("ReTruco") == 2
        assert get_truco_points_on_reject("ValeCuatro") == 3

    def test_get_truco_points_on_reject_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized truco stage: BadTruco"):
            get_truco_points_on_reject("BadTruco")


class TestEnvidoOrder:
    def test_envido_progression(self):
        envido_stages = valid_envido_order

        accept_points = []
        for stage in envido_stages[:-1]:
            accept_points.append(get_envido_points_on_accept(stage))
        assert accept_points == sorted(accept_points)

        reject_points = []
        for stage in envido_stages:
            reject_points.append(get_envido_points_on_reject(stage))
        assert reject_points == sorted(reject_points)

    def test_envido_order_length(self):
        assert len(valid_envido_order) == 3
        assert len(set(valid_envido_order)) == 3


class TestTrucoOrder:
    def test_truco_progression(self):
        truco_stages = valid_truco_order

        accept_points = []
        for stage in truco_stages:
            accept_points.append(get_truco_points_on_accept(stage))
        assert accept_points == sorted(accept_points)
        assert accept_points == [2, 3, 4]

        reject_points = []
        for stage in truco_stages:
            reject_points.append(get_truco_points_on_reject(stage))
        assert reject_points == sorted(reject_points)
        assert reject_points == [1, 2, 3]

    def test_truco_order_length(self):
        assert len(valid_truco_order) == 3
        assert len(set(valid_truco_order)) == 3


class TestPointRelationships:
    def test_envido_accept_greater_than_reject(self):
        for stage in ["Envido", "RealEnvido"]:
            accept = get_envido_points_on_accept(stage)
            reject = get_envido_points_on_reject(stage)
            assert accept > reject

        assert get_envido_points_on_accept("FaltaEnvido") == "game"
        assert get_envido_points_on_reject("FaltaEnvido") == 5

    def test_truco_accept_greater_than_reject(self):
        for stage in valid_truco_order:
            accept = get_truco_points_on_accept(stage)
            reject = get_truco_points_on_reject(stage)
            assert accept > reject

    def test_truco_reject_equals_previous_accept(self):
        assert get_truco_points_on_reject("ReTruco") == get_truco_points_on_accept("Truco")
        assert get_truco_points_on_reject("ValeCuatro") == get_truco_points_on_accept("ReTruco")