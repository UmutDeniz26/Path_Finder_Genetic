import sys
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

def exit_with_print(error, exit_code=0):
    """
    Print the message and exit the program
    
    Args:
        message (str): The message to print
        exit_code (int): The exit code
    
    Returns:
        None
    """
    print("An error occured: ", error)
    sys.exit(exit_code)