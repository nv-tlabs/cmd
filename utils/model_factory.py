"""Select model wrappers without leaking backend conditionals into trainers."""


def _family(args) -> str:
    family = getattr(args, "model_family", "wan").lower()
    if family not in {"wan", "cosmos"}:
        raise ValueError(f"Unsupported model_family: {family}")
    return family


def _wrapper_classes(args):
    if _family(args) == "cosmos":
        from cosmos import CosmosDiffusionWrapper, CosmosTextEncoder, CosmosVAEWrapper

        return CosmosDiffusionWrapper, CosmosTextEncoder, CosmosVAEWrapper

    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

    return WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def build_diffusion_wrapper(args, is_causal: bool, model_name=None):
    diffusion_wrapper, _, _ = _wrapper_classes(args)
    # Preserve the original Wan score-wrapper construction; Cosmos needs the
    # checkpoint filename shared by its causal and bidirectional variants.
    kwargs = (
        dict(getattr(args, "model_kwargs", {}))
        if model_name is None or _family(args) == "cosmos"
        else {}
    )
    if model_name is not None:
        kwargs["model_name"] = model_name
    return diffusion_wrapper(**kwargs, is_causal=is_causal)


def build_text_encoder(args):
    _, text_encoder, _ = _wrapper_classes(args)
    if _family(args) == "cosmos":
        return text_encoder(
            model_name=getattr(args, "text_encoder_name", "nvidia/Cosmos-Reason1-7B"),
            max_length=getattr(args, "text_encoder_max_length", 512),
        )
    return text_encoder()


def build_vae(args):
    _, _, vae = _wrapper_classes(args)
    if _family(args) == "cosmos":
        return vae(
            model_name=getattr(args, "vae_model_name", "nvidia/Cosmos-Predict2.5-2B"),
            checkpoint_filename=getattr(args, "vae_checkpoint_filename", "tokenizer.pth"),
        )
    return vae()
