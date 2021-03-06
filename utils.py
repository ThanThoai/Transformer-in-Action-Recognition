import time
from pytorch_lightning.utilities.distributed import rank_zero_only

@rank_zero_only
def print_on_rank_zero(content):
    print(content)
    

def timeit_wrapper(func, *args, **kwargs):
    start = time.perf_counter()
    func_return_val = func(*args, **kwargs)
    end = time.perf_counter()
    return func_return_val, float(f'{end - start:.4f}')



def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [
        38.7568578 / norm_value, 37.88248729 / norm_value,
        40.02898126 / norm_value
    ]


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value