import pandas as pd
import numpy as np
import time


def save_dataframe_hdf5(result_pd, save_lim, path, metadata):
    """
    Save the result dataframe to the given path with the given metadata
    
    Args:
        result_pd (pd.DataFrame): The dataframe to be saved
        save_lim (int): The limit of the dataframe to be saved
        path (str): The path to save the dataframe
        metadata (dict): The metadata to be saved with the dataframe
            metadata = {
                "LEARNING_RATE": learning_rate(float)
                "MUTATION_RATE": mutation_rate(float),
                "SELECT_PER_EPOCH": select_per_epoch(int),
                "MULTIPLIER": multiplier(int),
                "BOARD_SIZE": board_size(tuple),
                "EPOCH_COUNT": epoch_count(int),
            }

    Returns:
        None
    """

    save_lim = min(save_lim, len(result_pd))
    
    # if dict
    if isinstance(result_pd, dict):
        result_pd = pd.DataFrame(result_pd)[:save_lim] 

    elif isinstance(result_pd, pd.DataFrame):
        result_pd = result_pd[:save_lim]

    elif isinstance(result_pd, list):
        print("List object couldnt saved as dataframe")
        raise ValueError
    else:
        print("Unknown type")
        raise ValueError
    
    # Prepare the dataframe
    if 'Status' in result_pd.columns:
        result_pd = result_pd.drop( result_pd[result_pd['Status'] == 'initial'].index)
    
    # Save the dataframe
    store = pd.HDFStore(path)
    store.put('results', result_pd)

    store.get_storer('results').attrs.metadata = {
        "LEARNING_RATE": metadata["LEARNING_RATE"],
        "MUTATION_RATE": metadata["MUTATION_RATE"],
        "SELECT_PER_EPOCH": metadata["SELECT_PER_EPOCH"],
        "MULTIPLIER": metadata["MULTIPLIER"],
        "BOARD_SIZE": metadata["BOARD_SIZE"],
        "EPOCH_COUNT": metadata["EPOCH_COUNT"],
        "TIME": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    store.close()

def load_dataframe_hdf5(path):
    """
    Load the dataframe from the given path
    
    Args:
        path (str): The path to load the dataframe

    Returns:
        pd.DataFrame: The loaded dataframe
    """

    store = pd.HDFStore(path)
    result_pd = store['results']
    metadata = store.get_storer('results').attrs.metadata
    store.close()
    return result_pd, metadata