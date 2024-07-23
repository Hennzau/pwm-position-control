import numpy as np
import pyarrow as pa


def construct_pwm_to_logical_conversion_table_numpy(
    positions: (np.ndarray, np.ndarray),
    joints: list[str],
    targets: (np.ndarray, np.ndarray),
) -> {str: {str: int}}:
    """
    Constructs a conversion table from PWM positions to logical values using numpy.

    :param positions: A tuple of two numpy arrays representing the positions.
    :type positions: (np.ndarray, np.ndarray)
    :param joints: A list of joint names corresponding to the positions.
    :type joints: list[str]
    :param targets: A tuple of two numpy arrays representing the target values.
    :type targets: (np.ndarray, np.ndarray)

    :return: A dictionary with joints as keys and their respective conversion tables as values.
    :rtype: {str: {str: int}}

    Example:
    --------
    positions = (np.array([1024, 2048]), np.array([3072, 4096]))
    joints = ["joint1", "joint2"]
    targets = (np.array([10, 20]), np.array([30, 40]))
    conversion_table = construct_pwm_to_logical_conversion_table_numpy(positions, joints, targets)
    """

    result: {str: {str: int}} = {}

    for i in range(len(positions[0])):
        table: {str: int} = {}

        first, second = (
            round((positions[0][i] % 4096) / 1024) * 1024 % 4096,
            round((positions[1][i] % 4096) / 1024) * 1024 % 4096,
        )

        first, second = first * 360 / 4096, second * 360 / 4096

        first, second = int(first), int(second)

        target_first, target_second = targets[0][i], targets[1][i]

        table[str(first)] = target_first
        table[str(second)] = target_second

        for j in range(5):
            index = j * 90

            if index != first and index != second:
                if first < second:
                    offset = ((index - first) // 90) * (target_second - target_first)
                    table[str(index)] = target_first + offset

                    if table[str(index)] < -180:
                        table[str(index)] = table[str(index)] % 360
                    elif table[str(index)] > 180:
                        table[str(index)] = table[str(index)] % (-360)
                else:
                    offset = ((index - second) // 90) * (target_first - target_second)
                    table[str(index)] = target_second + offset

                    if table[str(index)] < -180:
                        table[str(index)] = table[str(index)] % 360
                    elif table[str(index)] > 180:
                        table[str(index)] = table[str(index)] % (-360)

        result[joints[i]] = table

    return result


def construct_pwm_to_logical_conversion_table_arrow(
    positions: (pa.StructArray, pa.StructArray),
    targets: (pa.StructArray, pa.StructArray),
) -> {str: {str: int}}:
    """
    Constructs a conversion table from PWM positions to logical values using pyarrow.

    :param positions: A tuple of two pyarrow structured arrays representing the positions.
    :type positions: (pa.StructArray, pa.StructArray)
    :param targets: A tuple of two pyarrow structured arrays representing the target values.
    :type targets: (pa.StructArray, pa.StructArray)

    :return: A dictionary with joints as keys and their respective conversion tables as values.
    :rtype: {str: {str: int}}
    """

    result: {str: {str: int}} = {}

    for i in range(len(positions[0].field("values"))):
        table: {str: int} = {}

        first, second = (
            round((positions[0].field("values")[i].as_py() % 4096) / 1024)
            * 1024
            % 4096,
            round((positions[1].field("values")[i].as_py() % 4096) / 1024)
            * 1024
            % 4096,
        )

        first, second = first * 360 / 4096, second * 360 / 4096

        first, second = int(first), int(second)

        target_first, target_second = (
            targets[0].field("values")[i].as_py(),
            targets[1].field("values")[i].as_py(),
        )

        table[str(first)] = target_first
        table[str(second)] = target_second

        for j in range(5):
            index = j * 90

            if index != first and index != second:
                if first < second:
                    offset = ((index - first) // 90) * (target_second - target_first)
                    table[str(index)] = target_first + offset

                    if table[str(index)] < -180:
                        table[str(index)] = table[str(index)] % 360
                    elif table[str(index)] > 180:
                        table[str(index)] = table[str(index)] % (-360)
                else:
                    offset = ((index - second) // 90) * (target_first - target_second)
                    table[str(index)] = target_second + offset

                    if table[str(index)] < -180:
                        table[str(index)] = table[str(index)] % 360
                    elif table[str(index)] > 180:
                        table[str(index)] = table[str(index)] % (-360)

        result[positions[0].field("joints")[i].as_py()] = table

    return result


def construct_logical_to_pwm_conversion_table_numpy(
    positions: (np.ndarray, np.ndarray),
    joints: list[str],
    targets: (np.ndarray, np.ndarray),
) -> {str: {str: int}}:
    """
    Constructs a conversion table from logical values to PWM positions using numpy.

    :param positions: A tuple of two numpy arrays representing the positions.
    :type positions: (np.ndarray, np.ndarray)
    :param joints: A list of joint names corresponding to the positions.
    :type joints: list[str]
    :param targets: A tuple of two numpy arrays representing the target values.
    :type targets: (np.ndarray, np.ndarray)

    :return: A dictionary with joints as keys and their respective conversion tables as values.
    :rtype: {str: {str: int}}

    Example:
    --------
    positions = (np.array([1024, 2048]), np.array([3072, 4096]))
    joints = ["joint1", "joint2"]
    targets = (np.array([10, 20]), np.array([30, 40]))
    conversion_table = build_logical_to_pwm_conversion_table_numpy(positions, joints, targets)
    """

    result: {str: {str: int}} = {}

    for i in range(len(positions[0])):
        table: {str: int} = {}

        first, second = (
            round(positions[0][i] / 1024) * 1024,
            round(positions[1][i] / 1024) * 1024,
        )

        first, second = first * 360 / 4096, second * 360 / 4096

        first, second = int(first), int(second)

        target_first, target_second = targets[0][i], targets[1][i]

        table[str(target_first)] = first
        table[str(target_second)] = second

        for j in range(5):
            index = j * 90

            if index != target_first and index != target_second:
                if target_first < target_second:
                    offset = ((index - target_first) // 90) * (second - first)
                    table[str(index)] = first + offset
                else:
                    offset = ((index - target_second) // 90) * (first - second)
                    table[str(index)] = second + offset

        result[joints[i]] = table

    return result


def construct_logical_to_pwm_conversion_table_arrow(
    positions: (pa.StructArray, pa.StructArray),
    targets: (pa.StructArray, pa.StructArray),
) -> {str: {str: int}}:
    """
    Constructs a conversion table from logical values to PWM positions using pyarrow.

    :param positions: A tuple of two pyarrow structured arrays representing the positions.
    :type positions: (pa.StructArray, pa.StructArray)
    :param targets: A tuple of two pyarrow structured arrays representing the target values.
    :type targets: (pa.StructArray, pa.StructArray)

    :return: A dictionary with joints as keys and their respective conversion tables as values.
    :rtype: {str: {str: int}}
    """

    result: {str: {str: int}} = {}

    for i in range(len(positions[0])):
        table: {str: int} = {}

        first, second = (
            round(positions[0].field("values")[i].as_py() / 1024) * 1024,
            round(positions[1].field("values")[i].as_py() / 1024) * 1024,
        )

        first, second = first * 360 / 4096, second * 360 / 4096

        first, second = int(first), int(second)

        target_first, target_second = (
            targets[0].field("values")[i].as_py(),
            targets[1].field("values")[i].as_py(),
        )

        table[str(target_first)] = first
        table[str(target_second)] = second

        for j in range(5):
            index = j * 90 - 180

            if index != target_first and index != target_second:
                if target_first < target_second:
                    offset = ((index - target_first) // 90) * (second - first)
                    table[str(index)] = first + offset
                else:
                    offset = ((index - target_second) // 90) * (first - second)
                    table[str(index)] = second + offset

        result[positions[0].field("joints")[i].as_py()] = table

    return result
