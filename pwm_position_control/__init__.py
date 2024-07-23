def select_limits_physical_to_logical(a, b):
    if a == 180 and b != 90 and b != 270:
        return -180, b
    if b == 180 and a != 90 and a != 270:
        return a, -180

    if a == -180 and b != -90 and b != -270:
        return 180, b

    if b == -180 and a != -90 and a != -270:
        return a, 180

    return a, b


def build_physical_to_logical(table: {str: int}):
    zero_to_ninety = select_limits_physical_to_logical(table["0"], table["90"])

    ninety_to_one_eighty = select_limits_physical_to_logical(table["90"], table["180"])

    one_eighty_to_two_seventy = select_limits_physical_to_logical(
        table["180"], table["270"]
    )

    two_seventy_to_zero = select_limits_physical_to_logical(table["270"], table["360"])

    def physical_to_logical_converter(x):
        if 0 <= x < 90:
            return zero_to_ninety[0] + (x / 90) * (
                zero_to_ninety[1] - zero_to_ninety[0]
            )

        if 90 <= x < 180:
            return ninety_to_one_eighty[0] + ((x - 90) / 90) * (
                ninety_to_one_eighty[1] - ninety_to_one_eighty[0]
            )

        if 180 <= x < 270:
            return one_eighty_to_two_seventy[0] + ((x - 180) / 90) * (
                one_eighty_to_two_seventy[1] - one_eighty_to_two_seventy[0]
            )

        if 270 <= x <= 360:
            return two_seventy_to_zero[0] + ((x - 270) / 90) * (
                two_seventy_to_zero[1] - two_seventy_to_zero[0]
            )

        else:
            return 0

    return physical_to_logical_converter


def build_logical_to_physical(table: {str: int}):
    minus_one_eighty_to_minus_ninety = (table["-180"], table["-90"])

    minus_ninety_to_zero = (table["-90"], table["0"])

    zero_to_ninety = (table["0"], table["90"])

    ninety_to_one_eighty = (table["90"], table["180"])

    def logical_to_physical_function(x):
        if -180 <= x < -90:
            return minus_one_eighty_to_minus_ninety[0] + ((x + 180) / 90) * (
                minus_one_eighty_to_minus_ninety[1]
                - minus_one_eighty_to_minus_ninety[0]
            )

        if -90 <= x < 0:
            return minus_ninety_to_zero[0] + ((x + 90) / 90) * (
                minus_ninety_to_zero[1] - minus_ninety_to_zero[0]
            )

        if 0 <= x < 90:
            return zero_to_ninety[0] + (x / 90) * (
                zero_to_ninety[1] - zero_to_ninety[0]
            )

        if 90 <= x <= 180:
            return ninety_to_one_eighty[0] + ((x - 90) / 90) * (
                ninety_to_one_eighty[1] - ninety_to_one_eighty[0]
            )

        return 0

    return logical_to_physical_function
