from models.attention_mil import AttentionMILModel, xAttentionMIL
from models.transmil import TransMIL, xTransMIL
from models.utils import ModelEngine


def get_model_and_classifier(
    model_type,
    num_features,
    num_classes,
    model_dims,
    dropout=False,
    n_out_layers=0,
    learning_rate=0.01,
    weight_decay=0.001,
    device="cpu",
):
    if model_type == "attention_mil":
        model = AttentionMILModel(
            input_dim=num_features,
            head_dim=num_classes,
            features_dim=model_dims,
            inner_attention_dim=model_dims,
            dropout=0.5 if dropout else 0,
            dropout_strategy="features",
            num_layers=1,
            n_out_layers=n_out_layers,
            bias=True,
            device=device,
        )
        classifier = ModelEngine(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_type="cross-entropy",
            gradient_clip=None,
            device=device,
        )
    elif model_type == "transmil":
        model = TransMIL(
            n_feat_input=num_features,
            n_feat=model_dims,
            head_dim=num_classes,
            device=device,
            attention="dot_prod",
            n_layers=2,
            dropout_att=0.5 if dropout else 0,
            dropout_class=0.5 if dropout else 0,
            dropout_feat=0.2 if dropout else 0,
            attn_residual=True,
            pool_method="cls_token",
            n_out_layers=n_out_layers,
            bias=True,
        )
        classifier = ModelEngine(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer="SGD",
            loss_type="cross-entropy",
            gradient_clip=None,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, classifier


def get_xmodel(model_type, explanation_type, model, detach_pe=False):
    explained_rel = "softmax" if explanation_type == "perturbation_keep" else "logit"
    if model_type == "attention_mil":
        xmodel = xAttentionMIL(
            model=model,
            explained_class=None,
            explained_rel=explained_rel,
            lrp_params=None,
            contrastive_class=None,
            detach_attn=True,
        )
    elif model_type == "transmil":
        xmodel = xTransMIL(
            model=model,
            explained_class=None,
            explained_rel=explained_rel,
            lrp_params=None,
            contrastive_class=None,
            discard_ratio=0,
            attention_layer=None,
            head_fusion="mean",
            detach_norm=None,
            detach_mean=False,
            detach_pe=detach_pe,
        )
    else:
        raise ValueError(
            f"No explanation class implemented for model of type: {type(model)}"
        )
    return xmodel
