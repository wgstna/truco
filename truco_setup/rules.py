from typing import Union

max_points: int = 30

envido_points_accept = {
    "Envido":      2,
    "RealEnvido":  5,
    "FaltaEnvido": "game",
}

envido_points_reject = {
    "Envido":      1,
    "RealEnvido":  3,
    "FaltaEnvido": 5
}

valid_envido_order = ["Envido", "RealEnvido", "FaltaEnvido"]

def get_envido_points_on_accept(stage: str) -> Union[int, str]:
    if stage not in envido_points_accept:
        raise ValueError(f"Unrecognized envido stage: {stage}")
    return envido_points_accept[stage]

def get_envido_points_on_reject(stage: str) -> int:
    if stage not in envido_points_reject:
        raise ValueError(f"Unrecognized envido stage: {stage}")
    return envido_points_reject[stage]

truco_points_accept = {
    "Truco":      2,
    "ReTruco":    3,
    "ValeCuatro": 4
}

truco_points_reject = {
    "Truco":      1,
    "ReTruco":    2,
    "ValeCuatro": 3
}

valid_truco_order = ["Truco", "ReTruco", "ValeCuatro"]

def get_truco_points_on_accept(stage: str) -> int:
    if stage not in truco_points_accept:
        raise ValueError(f"Unrecognized truco stage: {stage}")
    return truco_points_accept[stage]

def get_truco_points_on_reject(stage: str) -> int:
    if stage not in truco_points_reject:
        raise ValueError(f"Unrecognized truco stage: {stage}")
    return truco_points_reject[stage]