'''
Utility Functions for logging 

'''

import logging

# Logging
def define_logging(
      filename       : str,
    ) -> None        : 
    '''
    '''

    # Remove undesired logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename = filename,
        filemode = 'w',
        format   = '%(asctime)s - %(levelname)s - %(name)s : %(message)s',
        datefmt  = '%d-%b-%y %H:%M:%S',
        level    = logging.INFO
    )


    return

