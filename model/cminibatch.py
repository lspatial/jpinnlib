import numpy as np
import math


def iterate_minibatches(index_data, batch_size, shuffle=False):
    """
    A generator function that yields minibatches of indices from a dataset.

    This function allows for efficient batch-wise iteration through a dataset
    by returning subsets of indices that can be used to access the actual data.

    Parameters:
    -----------
    index_data : numpy.ndarray
        Array of indices representing the dataset samples.
    batch_size : int
        The size of each minibatch to yield.
    shuffle : bool, optional
        Whether to shuffle the indices before batching (default False).

    Yields:
    -------
    numpy.ndarray
        A subset of indices representing a single minibatch.

    Example:
    --------
    >>> indices = np.arange(100)
    >>> for batch_indices in iterate_minibatches(indices, batch_size=16, shuffle=True):
    >>>     batch_data = dataset[batch_indices]  # Access the actual data using indices
    >>>     # Process batch_data
    """
    indices_data = index_data.copy()  # Create a copy to avoid modifying the original

    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices_data)

    # Iterate through the indices in steps of batch_size
    for i, start_idx in enumerate(range(0, len(indices_data), batch_size)):
        end_idx = min(start_idx + batch_size, len(indices_data))  # Handle last batch which might be smaller
        excerpt_data = indices_data[start_idx:end_idx]  # Extract the batch indices
        yield excerpt_data  # Return this batch


def iterate_minibatches_traintest(index_train, index_test, index_indtest, batch_size, shuffle=False):
    """
    A generator function that yields synchronized minibatches from training, test,
    and independent test datasets.

    This function ensures that all three datasets are iterated through in a coordinated
    manner, with batch sizes for test and independent test sets proportionally scaled
    to maintain the same number of batches across all sets.

    Parameters:
    -----------
    index_train : numpy.ndarray
        Array of indices for training data.
    index_test : numpy.ndarray
        Array of indices for test data.
    index_indtest : numpy.ndarray
        Array of indices for independent test data.
    batch_size : int
        The batch size to use for the training data.
    shuffle : bool, optional
        Whether to shuffle the indices before batching (default False).

    Yields:
    -------
    tuple of numpy.ndarray
        A tuple containing (train_batch_indices, test_batch_indices, indtest_batch_indices),
        where each element is a subset of indices for the respective dataset.

    Notes:
    ------
    The batch sizes for test and independent test sets are calculated proportionally
    to maintain the same number of iterations as the training set.
    """
    # Create copies to avoid modifying the originals
    index_train1 = index_train.copy()
    index_test1 = index_test.copy()
    index_indtest1 = index_indtest.copy()

    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(index_train1)
        np.random.shuffle(index_test1)
        np.random.shuffle(index_indtest1)

    # Calculate proportional batch sizes for test and independent test sets
    # This ensures all datasets will have the same number of batches
    batch_size_test = math.ceil(batch_size * index_test1.shape[0] / index_train1.shape[0])
    batch_size_indtest = math.ceil(batch_size * index_indtest1.shape[0] / index_train1.shape[0])

    # Iterate through the indices
    for i, start_idx in enumerate(range(0, len(index_train1), batch_size)):
        # Extract training batch
        end_idx = min(start_idx + batch_size, len(index_train1))
        excerpt_train = index_train1[start_idx:end_idx]

        # Extract test batch using proportional batch size
        start_idx_t = i * batch_size_test
        end_idx_t = min(start_idx_t + batch_size_test, len(index_test1))
        excerpt_test = index_test1[start_idx_t:end_idx_t]

        # Extract independent test batch using proportional batch size
        start_idx_t = i * batch_size_indtest
        end_idx_t = min(start_idx_t + batch_size_indtest, len(index_indtest1))
        excerpt_indtest = index_indtest1[start_idx_t:end_idx_t]

        yield excerpt_train, excerpt_test, excerpt_indtest, None


def iterate_minibatches_traintest_f(index_train, index_test, index_indtest, index_semi, batch_size, shuffle=False):
    """
    An extended generator function that yields synchronized minibatches from training,
    test, independent test, and semi-supervised datasets.

    Similar to iterate_minibatches_traintest but includes a fourth dataset (semi-supervised)
    in the synchronized iteration.

    Parameters:
    -----------
    index_train : numpy.ndarray
        Array of indices for training data.
    index_test : numpy.ndarray
        Array of indices for test data.
    index_indtest : numpy.ndarray
        Array of indices for independent test data.
    index_semi : numpy.ndarray
        Array of indices for semi-supervised data.
    batch_size : int
        The batch size to use for the training data.
    shuffle : bool, optional
        Whether to shuffle the indices before batching (default False).

    Yields:
    -------
    tuple of numpy.ndarray
        A tuple containing (train_batch_indices, test_batch_indices,
        indtest_batch_indices, semi_batch_indices), where each element is
        a subset of indices for the respective dataset.

    Notes:
    ------
    This function is useful for semi-supervised learning scenarios where additional
    unlabeled or partially labeled data is available alongside the main datasets.
    The batch sizes for all non-training sets are calculated proportionally to maintain
    the same number of iterations as the training set.
    """
    # Create copies to avoid modifying the originals
    index_train1 = index_train.copy()
    index_test1 = index_test.copy()
    index_indtest1 = index_indtest.copy()
    index_semi1 = index_semi.copy()

    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(index_train1)
        np.random.shuffle(index_test1)
        np.random.shuffle(index_indtest1)
        np.random.shuffle(index_semi1)

    # Calculate proportional batch sizes for each non-training dataset
    batch_size_test = math.ceil(batch_size * index_test1.shape[0] / index_train1.shape[0])
    batch_size_indtest = math.ceil(batch_size * index_indtest1.shape[0] / index_train1.shape[0])
    batch_size_semi = math.ceil(batch_size * index_semi1.shape[0] / index_train1.shape[0])

    # Iterate through the indices
    for i, start_idx in enumerate(range(0, len(index_train1), batch_size)):
        # Extract training batch
        end_idx = min(start_idx + batch_size, len(index_train1))
        excerpt_train = index_train1[start_idx:end_idx]

        # Extract test batch using proportional batch size
        start_idx_t = i * batch_size_test
        end_idx_t = min(start_idx_t + batch_size_test, len(index_test1))
        excerpt_test = index_test1[start_idx_t:end_idx_t]

        # Extract independent test batch using proportional batch size
        start_idx_t = i * batch_size_indtest
        end_idx_t = min(start_idx_t + batch_size_indtest, len(index_indtest1))
        excerpt_indtest = index_indtest1[start_idx_t:end_idx_t]

        # Extract semi-supervised batch using proportional batch size
        start_idx_t = i * batch_size_semi
        end_idx_t = min(start_idx_t + batch_size_semi, len(index_semi1))
        excerpt_semi = index_semi1[start_idx_t:end_idx_t]

        yield excerpt_train, excerpt_test, excerpt_indtest, excerpt_semi