import torch
from . import model_eval_utils as eval_utils
from . import models
from captum.attr import IntegratedGradients

def compute_attributions(model, embeddings, task_labels, method='IG', single_label=None):
    # Initialize attribution method
    if method == 'IG':
        explainer = IntegratedGradients(model)
    else:
        raise ValueError(f"Unsupported attribution method: {method}")

    if single_label:
        attributions = explainer.attribute(embeddings, target=single_label)
    else:
        attributions_list = []
        for label_index in task_labels:
            label_attributions = explainer.attribute(embeddings, target=label_index)
            attributions_list.append(label_attributions)
        attributions = attributions_list
    
    return attributions

def compute_all_seed_attributions(
    modalities, 
    base_classifier_path, 
    task, 
    data,
    labels_interest=None,
    fusion_method="concatenation",
    modality_lambdas=None,
    baseline=False,
    args=None,
):
    num_modalities = len(modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality_dim = eval_utils._MODALITY_SHAPES["projection"]
    if task == "mortality":
        aggregate_attributions = [0] * num_modalities
    elif task == "phenotyping":
        aggregate_attributions = [[0] * num_modalities] * len(labels_interest)
    
    for seed_embed, seed in zip(data, eval_utils._SWEEP_SEEDS):
        if baseline:
            model = models.MultiModalBaseline(
                ts_input_dim=eval_utils._MODALITY_SHAPES["ts"],
                demo_input_dim=eval_utils._MODALITY_SHAPES["demo"],
                projection_dim=256,
                args=args,
                device=device,
            )
            seed_classifier_path = base_classifier_path.replace("_0_frozen_weighted", f"_{seed}_frozen_weighted")
            model.load_state_dict(torch.load(seed_classifier_path))
            model.to(device)
            classifier = model.classifier_head
        else:
            classifier = models.ClassificationHead(
                projection_dim=modality_dim,
                num_classes=eval_utils._TASK_CLASSES[task],
                fusion_method=fusion_method,
                num_modalities=num_modalities,
                modality_lambdas=modality_lambdas,
                verbose=False,
            )
            seed_classifier_path = base_classifier_path.replace("_0_frozen_weighted", f"_{seed}_frozen_weighted")
            classifier.load_state_dict(torch.load(seed_classifier_path))
            classifier.to(device)
        
        tensor_seed = torch.Tensor(seed_embed).to(device)
        tensor_seed.requires_grad = True
        
        if task == "mortality":
            attributions = compute_attributions(classifier, tensor_seed, labels_interest, single_label=1).cpu()
            reshaped_attributions = attributions.view(-1, num_modalities, modality_dim)
            mean_attributions = reshaped_attributions.abs().sum(dim=2)
            avg_mean_attributions = list(mean_attributions.mean(dim=0).detach().numpy())
            normed = avg_mean_attributions / sum(avg_mean_attributions)

            aggregate_attributions = [a + b for a, b in zip(aggregate_attributions, normed)]
        elif task == "phenotyping":
            attributions = compute_attributions(classifier, tensor_seed, labels_interest)
            for i, attribution in enumerate(attributions):
                attribution = attribution.cpu()
                reshaped_attribution = attribution.view(-1, num_modalities, modality_dim)
                mean_attributions = reshaped_attribution.abs().sum(dim=2)
                avg_mean_attribution = list(mean_attributions.mean(dim=0).detach().numpy())
                normed = avg_mean_attribution / sum(avg_mean_attribution)
                aggregate_attributions[i] = [a + b for a, b in zip(aggregate_attributions[i], normed)]
    
    if task == "mortality":
        global_mean_attributions = [float(a / 10) for a in aggregate_attributions]
    elif task == "phenotyping":
        global_mean_attributions = [[float(a / 10) for a in pheno_aggregate] for pheno_aggregate in aggregate_attributions]
    return global_mean_attributions