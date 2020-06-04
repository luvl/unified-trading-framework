import argparse

def parse_args():
    """Parse arguments."""
    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    parser = argparse.ArgumentParser(
        description="Python implementation of Master thesis 'A unified Machine Learning / Deep Learning framework for security trading and predicting traders’ strategies'"
    )
    parser.add_argument('--with_extended_feature', type=str_to_bool, default=False, help='add extended feature to observation')

    args = parser.parse_args()
    return args

