from math import isclose

from template import (
    duration_to_minutes,
    media,
    mediana,
    quantil,
    quartis,
    resumo_duracoes,
)


def assert_close(actual, expected):
    assert isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_duration_to_minutes():
    assert_close(duration_to_minutes("3:30"), 3.5)
    assert_close(duration_to_minutes("4:05"), 4 + 5 / 60)


def test_media_e_mediana():
    assert_close(media([1, 2, 3, 4]), 2.5)
    assert_close(mediana([1, 2, 99]), 2)
    assert_close(mediana([1, 2, 3, 4]), 2.5)


def test_quantil_e_quartis():
    dados = [1, 2, 3, 4, 5]
    assert_close(quantil(dados, 0.25), 2)
    assert_close(quantil(dados, 0.50), 3)
    assert_close(quantil(dados, 0.75), 4)
    assert quartis(dados) == (2, 3, 4)


def test_resumo_duracoes():
    resumo = resumo_duracoes(["3:00", "4:30", "2:30", "5:00"])

    assert resumo["n"] == 4
    assert_close(resumo["media"], 3.75)
    assert_close(resumo["mediana"], 3.75)
    assert_close(resumo["q1"], 2.875)
    assert_close(resumo["q3"], 4.625)
    assert_close(resumo["intervalo"], 2.5)
