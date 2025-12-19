#----------------------------------------------------------------------------
#
#      DM-TOA augmentation strategy
#
#----------------------------------------------------------------------------

import numpy as np

from typing import List, Union


def dm_filtering(
        toa_dm : np.ndarray,
        *,
        threshold : float,
    ):

    if toa_dm.shape[0] == 0:
        return toa_dm

    dm_flag = toa_dm[:, 1] >= threshold
    filtered_samples = toa_dm[dm_flag]

    return filtered_samples


def dm_shift(
        toa_dm : np.ndarray,
        *,
        shifts : Union[float, List[float]],
        trigger_dm : float = -1,
        min_dm : float = 0.0,
        max_dm : float = 6000.0
    ):

    if toa_dm.shape[0] == 0:
        return toa_dm

    if isinstance(shifts, float):
        shifts = [shifts]

    triggered_index = toa_dm[:, 1] >= trigger_dm
    augment_samples = toa_dm[triggered_index]

    if augment_samples.shape[0] == 0:
        return toa_dm

    output = [toa_dm]

    for shift in shifts:
        new_sample = augment_samples.copy()
        new_dm = augment_samples[:, 1] + shift
        new_dm = np.clip(new_dm, a_min=min_dm, a_max=max_dm)
        new_sample[:, 1] = new_dm

        output.append(new_sample)

    return np.concatenate(output, axis=0)


def toa_shift(
        toa_dm: np.ndarray,
        *,
        shifts: Union[float, List[float]],
        min_toa: float = 0.0,
        max_toa: float = 12.88
):
    if toa_dm.shape[0] == 0:
        return toa_dm

    if isinstance(shifts, float):
        shifts = [shifts]

    augment_samples = toa_dm

    if augment_samples.shape[0] == 0:
        return toa_dm

    output = [toa_dm]

    for shift in shifts:
        new_sample = augment_samples.copy()
        new_toa = augment_samples[:, 0] + shift
        new_toa = np.clip(new_toa, a_min=min_toa, a_max=max_toa)
        new_sample[:, 0] = new_toa

        output.append(new_sample)

    return np.concatenate(output, axis=0)


def entity(toa_dm : np.ndarray):

    return toa_dm
