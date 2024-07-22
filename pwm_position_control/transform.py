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


def physical_to_ranged_logical_numpy(
    physical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
) -> np.ndarray:
    """
    Converts physical values to ranged logical values using numpy.

    :param physical_values: A numpy array of physical values to be converted.
    :type physical_values: np.ndarray
    :param joints: A list of joint names corresponding to the physical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A numpy array of ranged logical values.
    :rtype: np.ndarray

    :raises ValueError: If physical_values contains None values or if lengths of joints and physical_values do not match.
    """

    if np.any(physical_values == None):
        raise ValueError("physical_position: np.ndarray cannot contain None values")

    if len(joints) != len(physical_values):
        raise ValueError("joints and physical_values must have the same length")

    return np.array(
        [
            table[joints[i]]["physical_to_logical"](physical_values[i] * 360 / 4096)
            for i in range(len(joints))
        ],
        np.float32,
    )


def physical_to_ranged_logical_arrow(
    physical_values: pa.StructArray, table: {str: {str: Callable[[float], float]}}
) -> pa.StructArray:
    """
    Converts physical values to ranged logical values using pyarrow.

    :param physical_values: A structured array containing the physical values to be converted.
    :type physical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A structured array with joints and their corresponding ranged logical values.
    :rtype: pa.StructArray
    """

    joints = physical_values.field("joints")
    positions = physical_values.field("values")

    return wrap_joints_and_values(
        joints,
        pa.array(
            [
                table[joints[i].as_py()]["physical_to_logical"](
                    positions[i].as_py() * 360 / 4096
                )
                for i in range(len(joints))
            ],
            type=pa.float32(),
        ),
    )


def logical_to_physical_numpy(
    logical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
) -> np.ndarray:
    """
    Converts logical values to physical values using numpy.

    :param logical_values: A numpy array of logical values to be converted.
    :type logical_values: np.ndarray
    :param joints: A list of joint names corresponding to the logical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A numpy array of physical values.
    :rtype: np.ndarray

    :raises ValueError: If logical_values contains None values or if lengths of joints and logical_values do not match.
    """

    if np.any(logical_values == None):
        raise ValueError("logical_position: np.ndarray cannot contain None values")

    if len(joints) != len(logical_values):
        raise ValueError("joints and logical_position must have the same length")

    return np.array(
        [
            int(table[joints[i]]["logical_to_physical"](logical_values[i]) * 4096 / 360)
            for i in range(len(joints))
        ],
        np.int32,
    )


def logical_to_physical_arrow(
    logical_values: pa.StructArray, table: {str: {str: Callable[[float], float]}}
) -> pa.StructArray:
    """
    Converts logical values to physical values using pyarrow.

    :param logical_values: A structured array containing the logical values to be converted.
    :type logical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A structured array with joints and their corresponding physical values.
    :rtype: pa.StructArray
    """

    joints = logical_values.field("joints")
    positions = logical_values.field("values")

    return wrap_joints_and_values(
        joints,
        pa.array(
            [
                int(
                    table[joints[i].as_py()]["logical_to_physical"](
                        positions[i].as_py()
                    )
                    * 4096
                    / 360
                )
                for i in range(len(joints))
            ],
            type=pa.int32(),
        ),
    )


def physical_to_un_ranged_logical_numpy(
    physical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
) -> np.ndarray:
    """
    Converts physical values to un ranged logical values using numpy.

    :param physical_values: A numpy array of physical values to be converted.
    :type physical_values: np.ndarray
    :param joints: A list of joint names corresponding to the physical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A numpy array of un ranged logical values.
    :rtype: np.ndarray

    :raises ValueError: If physical_values contains None values or if lengths of joints and physical_values do not match.
    """

    if np.any(physical_values == None):
        raise ValueError("physical_position: np.ndarray cannot contain None values")

    if len(joints) != len(physical_values):
        raise ValueError("joints and physical_values must have the same length")

    base = logical_to_physical_numpy(
        physical_to_ranged_logical_numpy(physical_values, joints, table), joints, table
    )

    offset = np.array((physical_values - base) * 360 / 4096, dtype=np.float32)

    logical = physical_to_ranged_logical_numpy(physical_values, joints, table)

    return logical - offset


def physical_to_un_ranged_logical_arrow(
    physical_values: pa.StructArray, table: {str: {str: Callable[[float], float]}}
) -> pa.StructArray:
    """
    Converts physical values to un ranged logical values using pyarrow.

    :param physical_values: A structured array containing the physical values to be converted.
    :type physical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A structured array with joints and their corresponding un ranged logical values.
    :rtype: pa.StructArray
    """

    joints = physical_values.field("joints")
    positions = physical_values.field("values")

    base = logical_to_physical_arrow(
        physical_to_ranged_logical_arrow(physical_values, table), table
    )

    offset = pa.multiply(
        pa.subtract(positions, base.field("values")),
        pa.array([360 / 4096] * len(joints), type=pa.float32()),
    )

    logical = physical_to_ranged_logical_arrow(physical_values, table)

    return wrap_joints_and_values(
        joints,
        pa.subtract(logical.field("values"), offset),
    )


def logical_to_physical_with_offset_numpy(
    physical_values: np.ndarray,
    logical_values: np.ndarray,
    joints: list[str],
    table: {str: {str: Callable[[float], float]}},
) -> np.ndarray:
    """
    Converts logical values to physical values with an offset using numpy.

    :param physical_values: A numpy array of current physical values.
    :type physical_values: np.ndarray
    :param logical_values: A numpy array of target logical values.
    :type logical_values: np.ndarray
    :param joints: A list of joint names corresponding to the physical and logical values.
    :type joints: list[str]
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A numpy array of physical values adjusted with the offset.
    :rtype: np.ndarray

    :raises ValueError: If physical_values or logical_values contain None values or if lengths of joints and values do not match.
    """

    if np.any(physical_values == None):
        raise ValueError("physical_position: np.ndarray cannot contain None values")

    if len(joints) != len(physical_values):
        raise ValueError("joints and physical_values must have the same length")

    if np.any(logical_values == None):
        raise ValueError("logical_position: np.ndarray cannot contain None values")

    if len(joints) != len(logical_values):
        raise ValueError("joints and logical_position must have the same length")

    physical_ranged_goal = logical_to_physical_numpy(logical_values, joints, table)
    base = logical_to_physical_numpy(
        physical_to_un_ranged_logical_numpy(physical_values, joints, table),
        joints,
        table,
    )

    return physical_values - base + physical_ranged_goal


def logical_to_physical_with_offset_arrow(
    physical_values: pa.StructArray,
    logical_values: pa.StructArray,
    table: {str: {str: Callable[[float], float]}},
) -> pa.StructArray:
    """
    Converts logical values to physical values with an offset using pyarrow.

    :param physical_values: A structured array containing the current physical values.
    :type physical_values: pa.StructArray
    :param logical_values: A structured array containing the target logical values.
    :type logical_values: pa.StructArray
    :param table: A dictionary containing conversion functions for each joint.
    :type table: {str: {str: Callable[[float], float]}}

    :return: A structured array with joints and their corresponding physical values adjusted with the offset.
    :rtype: pa.StructArray
    """

    joints = physical_values.field("joints")

    physical_ranged_goal = logical_to_physical_arrow(logical_values, table)
    base = logical_to_physical_arrow(
        physical_to_un_ranged_logical_arrow(physical_values, table), table
    )

    return wrap_joints_and_values(
        joints,
        pa.add(
            pa.subtract(physical_values.field("values"), base.field("values")),
            physical_ranged_goal.field("values"),
        ),
    )
