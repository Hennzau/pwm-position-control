from typing import Callable, Union

import pyarrow as pa
import pyarrow.compute as pc

import numpy as np


def wrap_joints_and_values(
    joints: Union[list[str], np.array, pa.Array],
    values: Union[int, list[int], np.array, pa.Array],
) -> pa.StructArray:
    """
    Wraps joints and their corresponding values into a structured array.

    :param joints: A list, numpy array, or pyarrow array of joint names.
    :type joints: Union[list[str], np.array, pa.Array]
    :param values: A single integer value, or a list, numpy array, or pyarrow array of integer values.
                   If a single integer is provided, it will be broadcasted to all joints.
    :type values: Union[int, list[int], np.array, pa.Array]

    :return: A structured array with two fields:
             - "joints": A string field containing the names of the joints.
             - "values": An Int32Array containing the values corresponding to the joints.
    :rtype: pa.StructArray

    Example:
    --------
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    values = [100, 200, 300]
    struct_array = wrap_joints_and_values(joints, values)

    This example wraps the given joints and their corresponding values into a structured array.

    Another example with a single integer value:
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    value = 150
    struct_array = wrap_joints_and_values(joints, value)

    This example broadcasts the single integer value to all joints and wraps them into a structured array.
    """
    if isinstance(values, int):
        values = [values] * len(joints)

    return pa.StructArray.from_arrays(
        arrays=[joints, values],
        names=["joints", "values"],
    )


def pwm_to_logical_numpy(
    pwm_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
    ranged=True,
) -> np.ndarray:
    """
    Converts pwm values to ranged logical values using numpy.

    :param pwm_values: A numpy array of pwm values to be converted.
    :type pwm_values: np.ndarray
    :param joints: A list of joint names corresponding to the pwm values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}
    :param ranged: A boolean flag to determine if the logical values should be in the range [-180, 180].
    :type ranged: bool

    :return: A numpy array of ranged logical values.
    :rtype: np.ndarray

    :raises ValueError: If pwm_values contains None values or if lengths of joints and pwm_values do not match.
    """

    if np.any(pwm_values == None):
        raise ValueError("pwm_position: np.ndarray cannot contain None values")

    if len(joints) != len(pwm_values):
        raise ValueError("joints and pwm_values must have the same length")

    if ranged:
        return np.array(
            [
                table[joints[i]]["pwm_to_logical"]((pwm_values[i] * 360 / 4096) % 360)
                for i in range(len(joints))
            ],
            np.float32,
        )
    else:
        return np.array(
            [
                table[joints[i]]["pwm_to_logical"]((pwm_values[i] * 360 / 4096))
                for i in range(len(joints))
            ],
            np.float32,
        )


def pwm_to_logical_arrow(
    pwm_values: pa.StructArray,
    table: {str: {str: Callable[[float], float]}},
    ranged=True,
) -> pa.StructArray:
    """
    Converts pwm values to ranged logical values using pyarrow.

    :param pwm_values: A structured array containing the pwm values to be converted.
    :type pwm_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}
    :param ranged: A boolean flag to determine if the logical values should be in the range [-180, 180].
    :type ranged: bool

    :return: A structured array with joints and their corresponding ranged logical values.
    :rtype: pa.StructArray
    """

    joints = pwm_values.field("joints")
    positions = pwm_values.field("values")

    if ranged:
        return wrap_joints_and_values(
            joints,
            pa.array(
                [
                    table[joints[i].as_py()]["pwm_to_logical"](
                        (positions[i].as_py() * 360 / 4096) % 360
                    )
                    for i in range(len(joints))
                ],
                type=pa.float32(),
            ),
        )
    else:
        return wrap_joints_and_values(
            joints,
            pa.array(
                [
                    table[joints[i].as_py()]["pwm_to_logical"](
                        (positions[i].as_py() * 360 / 4096)
                    )
                    for i in range(len(joints))
                ],
                type=pa.float32(),
            ),
        )


def logical_to_pwm_numpy(
    logical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
    ranged: bool = True,
) -> np.ndarray:
    """
    Converts logical values to pwm values using numpy.

    :param logical_values: A numpy array of logical values to be converted.
    :type logical_values: np.ndarray
    :param joints: A list of joint names corresponding to the logical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}
    :param ranged: A boolean flag to determine if the pwm values should be calculated from the ranged logical values.
    :type ranged: bool

    :return: A numpy array of pwm values.
    :rtype: np.ndarray

    :raises ValueError: If logical_values contains None values or if lengths of joints and logical_values do not match.
    """

    if np.any(logical_values == None):
        raise ValueError("logical_position: np.ndarray cannot contain None values")

    if len(joints) != len(logical_values):
        raise ValueError("joints and logical_position must have the same length")

    if ranged:
        return np.array(
            [
                int(
                    table[joints[i]]["logical_to_pwm"](
                        (logical_values[i] + 180) % 360 - 180
                    )
                    * 4096
                    / 360
                )
                for i in range(len(joints))
            ],
            np.int32,
        )
    else:
        return np.array(
            [
                int(table[joints[i]]["logical_to_pwm"](logical_values[i]) * 4096 / 360)
                for i in range(len(joints))
            ],
            np.int32,
        )


def logical_to_pwm_arrow(
    logical_values: pa.StructArray,
    table: {str: {str: Callable[[float], float]}},
    ranged=True,
) -> pa.StructArray:
    """
    Converts logical values to pwm values using pyarrow.

    :param logical_values: A structured array containing the logical values to be converted.
    :type logical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}
    :param ranged: A boolean flag to determine if the pwm values should be calculated from the ranged logical values.
    :type ranged: bool

    :return: A structured array with joints and their corresponding pwm values.
    :rtype: pa.StructArray
    """

    joints = logical_values.field("joints")
    positions = logical_values.field("values")

    if ranged:
        return wrap_joints_and_values(
            joints,
            pa.array(
                [
                    int(
                        table[joints[i].as_py()]["logical_to_pwm"](
                            (positions[i].as_py() + 180) % 360 - 180
                        )
                        * 4096
                        / 360
                    )
                    for i in range(len(joints))
                ],
                type=pa.int32(),
            ),
        )
    else:
        return wrap_joints_and_values(
            joints,
            pa.array(
                [
                    int(
                        table[joints[i].as_py()]["logical_to_pwm"](positions[i].as_py())
                        * 4096
                        / 360
                    )
                    for i in range(len(joints))
                ],
                type=pa.int32(),
            ),
        )


def logical_to_pwm_with_offset_numpy(
    pwm_values: np.ndarray,
    logical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
) -> np.ndarray:
    """
    Converts logical values to pwm values with an offset using numpy.

    :param pwm_values: A numpy array of current pwm values.
    :type pwm_values: np.ndarray
    :param logical_values: A numpy array of target logical values.
    :type logical_values: np.ndarray
    :param joints: A list of joint names corresponding to the pwm and logical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A numpy array of pwm values adjusted with the offset.
    :rtype: np.ndarray

    :raises ValueError: If pwm_values or logical_values contain None values or if lengths of joints and values do not match.
    """

    if np.any(pwm_values == None):
        raise ValueError("pwm_position: np.ndarray cannot contain None values")

    if len(joints) != len(pwm_values):
        raise ValueError("joints and pwm_values must have the same length")

    if np.any(logical_values == None):
        raise ValueError("logical_position: np.ndarray cannot contain None values")

    if len(joints) != len(logical_values):
        raise ValueError("joints and logical_position must have the same length")

    pwm_ranged_goal = logical_to_pwm_numpy(logical_values, joints, table)
    base = logical_to_pwm_numpy(
        pwm_to_logical_numpy(pwm_values, joints, table),
        joints,
        table,
    )

    return pwm_values - base + pwm_ranged_goal


def logical_to_pwm_with_offset_arrow(
    pwm_values: pa.StructArray,
    logical_values: pa.StructArray,
    table: {str: {str: Callable[[float], float]}},
) -> pa.StructArray:
    """
    Converts logical values to pwm values with an offset using pyarrow.

    :param pwm_values: A structured array containing the current pwm values.
    :type pwm_values: pa.StructArray
    :param logical_values: A structured array containing the target logical values.
    :type logical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A structured array with joints and their corresponding pwm values adjusted with the offset.
    :rtype: pa.StructArray
    """

    joints = pwm_values.field("joints")

    pwm_ranged_goal = logical_to_pwm_arrow(logical_values, table)
    base = logical_to_pwm_arrow(pwm_to_logical_arrow(pwm_values, table), table)

    return wrap_joints_and_values(
        joints,
        pa.add(
            pa.subtract(pwm_values.field("values"), base.field("values")),
            pwm_ranged_goal.field("values"),
        ),
    )
