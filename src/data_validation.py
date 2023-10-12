import os
import pandas as pd

from . import utils

def check_data_types(input_data: pd.DataFrame, config: dict):
    # Check column data types
    assert input_data.select_dtypes('int').columns.to_list() == \
        config['int_columns'], 'an error occurs in int column(s).'
    
def check_data_range(input_data: pd.DataFrame, config: dict):
    # Check range of age
    assert input_data[config['int_columns'][0]].between(
        config['range_age'][0],
        config['range_age'][1]
    ).sum() == len(input_data), 'an error occurs in range_age.'

    # Check range of balance
    assert input_data[config['int_columns'][1]].between(
        config['range_balance'][0],
        config['range_balance'][1]
    ).sum() == len(input_data), 'an error occurs in range_balance.'

    # Check range of day
    assert input_data[config['int_columns'][2]].between(
        config['range_day'][0],
        config['range_day'][1]
    ).sum() == len(input_data), 'an error occurs in range_day.'

    # Check range of duration
    assert input_data[config['int_columns'][3]].between(
        config['range_duration'][0],
        config['range_duration'][1]
    ).sum() == len(input_data), 'an error occurs in range_duration.'

    # Check range of campaign
    assert input_data[config['int_columns'][4]].between(
        config['range_campaign'][0],
        config['range_campaign'][1]
    ).sum() == len(input_data), 'an error occurs in range_campaign.'

    # Check range of pdays
    assert input_data[config['int_columns'][5]].between(
        config['range_pdays'][0],
        config['range_pdays'][1]
    ).sum() == len(input_data), 'an error occurs in range_pdays.'

    # Check range of previous
    assert input_data[config['int_columns'][6]].between(
        config['range_previous'][0],
        config['range_previous'][1]
    ).sum() == len(input_data), 'an error occurs in range_previous.'

if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)

    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read raw dataset
    raw_dataset = pd.read_csv(
        os.path.join(file_dir, config['dataset_path']),
        delimiter=config['dataset_delimiter']
    )

    # 3. Check data types
    check_data_types(raw_dataset, config)

    # 4. Check data range
    check_data_range(raw_dataset, config)