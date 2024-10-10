from training.data_set_from_scanned_exams import digit


def test_first_digit_of_two_digit_number() -> None:
    assert digit(0, 12.5) == 1


def test_second_digit_of_two_digit_number() -> None:
    assert digit(1, 12.5) == 2


def test_third_digit_of_two_digit_number() -> None:
    assert digit(2, 12.5) == 5


def test_first_digit_of_one_digit_number_is_leading_zero() -> None:
    assert digit(0, 2.5) == 0


# no premature support for three digit sums
# def test_first_digit_of_three_digit_number() -> None:
#     assert digit(0, 123.5) == 1
#
#
# def test_fourth_digit_of_three_digit_number_is_implicit_zero() -> None:
#     assert digit(0, 123) == 0