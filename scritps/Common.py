def print_dict(dictionary):
    """
    Print the dictionary in a pretty way
    
    Args:
        dictionary (dict): The dictionary to print
    
    Returns:
        None
    """
    for key, value in dictionary.items():
        print(f"{key}: {value}")