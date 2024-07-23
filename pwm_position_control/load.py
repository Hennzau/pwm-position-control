from .functions import construct_control_table


def load_control_table_from_json_conversion_tables(
    tables: {str: {str: {str: int}}}, json_control_table: {}
):
    pwm_to_logical_table = {}
    logical_to_pwm_table = {}

    for joint in tables.keys():
        pwm_to_logical_table[joint] = tables[joint]["pwm_to_logical"]
        logical_to_pwm_table[joint] = tables[joint]["logical_to_pwm"]

    control_table = construct_control_table(pwm_to_logical_table, logical_to_pwm_table)

    for joint in json_control_table.keys():
        json_control_table[joint]["pwm_to_logical"] = control_table[joint][
            "pwm_to_logical"
        ]
        json_control_table[joint]["logical_to_pwm"] = control_table[joint][
            "logical_to_pwm"
        ]

    return control_table
