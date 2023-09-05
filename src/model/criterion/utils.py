import torch


def convert_to_multiclass_targets(
    mixed_targets, mixed_indices, multilabel_labels, multiclass_labels
):
    """
    Convert the mixed targets to multiclass format.

    Args:
    - mixed_targets: Tensor of shape [batch_size, len(mixed_labels)]
    - multilabel_labels: Dictionary mapping from multilabel category names to indices
    - multiclass_labels: Dictionary mapping from multiclass category names to indices

    Returns:
    - multiclass_targets: Tensor of shape [batch_size, len(multiclass_labels)]
    """
    batch_size, _ = mixed_targets.shape
    multiclass_targets = torch.zeros((batch_size, len(multiclass_labels)))

    # Loop through all the mixed labels and map them to multiclass indices
    for mixed_index, category in mixed_indices.items():
        if category in multiclass_labels:
            multiclass_index = multiclass_labels[category]
            if category in multilabel_labels:
                multiclass_targets[:, multiclass_index] = 0.5
            else:
                multiclass_targets[:, multiclass_index] = mixed_targets[:, mixed_index]

    return multiclass_targets


def convert_to_multilabel_targets(mixed_targets, mixed_indices, multilabel_labels):
    """
    Convert the mixed targets to multilabel format.

    Args:
    - mixed_targets: Tensor of shape [batch_size, 163]
    - multilabel_labels: Dictionary mapping from multilabel category names to indices

    Returns:
    - multilabel_targets: Tensor of shape [batch_size, len(multilabel_labels)]
    """
    batch_size, _ = mixed_targets.shape
    multilabel_targets = torch.zeros((batch_size, len(multilabel_labels)))

    # Loop through all the mixed labels and map them to multilabel indices
    for mixed_index, category in mixed_indices.items():
        if category in multilabel_labels:
            multilabel_index = multilabel_labels[category]
            multilabel_targets[:, multilabel_index] = mixed_targets[:, mixed_index]

    return multilabel_targets


def convert_targets(
    targets, mixed_indices, multilabel_labels, multiclass_labels, dataset_indices
):
    multiclass_targets = convert_to_multiclass_targets(
        targets, mixed_indices, multilabel_labels, multiclass_labels
    )
    multilabel_targets = convert_to_multilabel_targets(
        targets, mixed_indices, multilabel_labels
    )

    return multiclass_targets, multilabel_targets
