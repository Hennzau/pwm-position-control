import copy
from typing import Callable


def construct_pwm_to_logical_control_table(
    table: {str: {str: int}},
) -> {str: Callable}:
    """
    Constructs a control table that maps PWM values to logical values.

    :param table: A dictionary with joint names as keys and dictionaries as values,
                  where the inner dictionaries map angles (as strings) to integer values.
    :type table: {str: {str: int}}

    :return: A dictionary with joint names as keys and callables as values.
             The callables convert physical PWM values to logical values.
    :rtype: {str: Callable}

    Example:
    --------
    table = {
        "joint1": {"0": 0, "90": 90, "180": 180, "270": 270, "360": 360},
        "joint2": {"0": 0, "90": 90, "180": 180, "270": 270, "360": 360},
    }
    control_table = construct_pwm_to_logical_control_table(table)
    """

    def select_limits_pwm_to_logical(a, b):
        if a == 180 and b != 90 and b != 270:
            return -180, b
        if b == 180 and a != 90 and a != 270:
            return a, -180

        if a == -180 and b != -90 and b != -270:
            return 180, b

        if b == -180 and a != -90 and a != -270:
            return a, 180

        return a, b

    def create_function_for_joint(joint, values, control_table):
        zero_to_ninety = select_limits_pwm_to_logical(values["0"], values["90"])

        ninety_to_one_eighty = select_limits_pwm_to_logical(values["90"], values["180"])

        one_eighty_to_two_seventy = select_limits_pwm_to_logical(
            values["180"], values["270"]
        )

        two_seventy_to_zero = select_limits_pwm_to_logical(values["270"], values["360"])

        def _f(x):
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

        control_table[joint] = _f

    control_table: {str: Callable} = {}

    for joint, values in table.items():
        create_function_for_joint(joint, values, control_table)

    return control_table


def construct_logical_to_pwm_control_table(
    table: {str: {str: int}},
) -> {str: Callable}:
    """
    Constructs a control table that maps logical values to PWM values.

    :param table: A dictionary with joint names as keys and dictionaries as values,
                  where the inner dictionaries map angles (as strings) to integer values.
    :type table: {str: {str: int}}

    :return: A dictionary with joint names as keys and callables as values.
             The callables convert logical values to physical PWM values.
    :rtype: {str: Callable}

    Example:
    --------
    table = {
        "joint1": {"-180": -180, "-90": -90, "0": 0, "90": 90, "180": 180},
        "joint2": {"-180": -180, "-90": -90, "0": 0, "90": 90, "180": 180},
    }
    control_table = construct_logical_to_pwm_control_table(table)
    """

    def create_function_for_joint(joint, values, control_table):
        minus_one_eighty_to_minus_ninety = (values["-180"], values["-90"])

        minus_ninety_to_zero = (values["-90"], values["0"])

        zero_to_ninety = (values["0"], values["90"])

        ninety_to_one_eighty = (values["90"], values["180"])

        def logical_to_physical_converter(x):
            if x < -90:
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

            if 90 <= x:
                return ninety_to_one_eighty[0] + ((x - 90) / 90) * (
                    ninety_to_one_eighty[1] - ninety_to_one_eighty[0]
                )

            else:
                return 0

        control_table[joint] = logical_to_physical_converter

    control_table: {str: Callable} = {}

    for joint, values in table.items():
        create_function_for_joint(joint, values, control_table)

    return control_table


def construct_control_table(
    pwm_to_logical_table: {str: {str: int}},
    logical_to_pwm_table: {str: {str: int}},
) -> {str: {str: Callable}}:
    """
    Constructs a comprehensive control table that includes both PWM-to-logical
    and logical-to-PWM conversion tables.

    :param pwm_to_logical_table: A dictionary with joint names as keys and dictionaries as values,
                                 where the inner dictionaries map PWM angles (as strings) to integer logical values.
    :type pwm_to_logical_table: {str: {str: int}}
    :param logical_to_pwm_table: A dictionary with joint names as keys and dictionaries as values,
                                 where the inner dictionaries map logical angles (as strings) to integer PWM values.
    :type logical_to_pwm_table: {str: {str: int}}

    :return: A dictionary with two keys: "pwm_to_logical" and "logical_to_pwm".
             Each key maps to a dictionary with joint names as keys and callables as values
             for converting between physical PWM values and logical values.
    :rtype: {str: {str: Callable}}

    Example:
    --------
    pwm_to_logical_table = {
        "joint1": {"0": 0, "90": 90, "180": 180, "270": 270, "360": 360},
        "joint2": {"0": 0, "90": 90, "180": 180, "270": 270, "360": 360},
    }
    logical_to_pwm_table = {
        "joint1": {"-180": -180, "-90": -90, "0": 0, "90": 90, "180": 180},
        "joint2": {"-180": -180, "-90": -90, "0": 0, "90": 90, "180": 180},
    }
    control_table = construct_control_table(pwm_to_logical_table, logical_to_pwm_table)
    """

    pwm_to_logical_control_table = construct_pwm_to_logical_control_table(
        pwm_to_logical_table
    )

    logical_to_pwm_control_table = construct_logical_to_pwm_control_table(
        logical_to_pwm_table
    )

    return {
        joint: {
            "pwm_to_logical": pwm_to_logical_control_table[joint],
            "logical_to_pwm": logical_to_pwm_control_table[joint],
        }
        for joint in list(pwm_to_logical_table.keys())
    }
