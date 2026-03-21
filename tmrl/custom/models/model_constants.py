# TODO: Verify these values against literature
# Aligning with scientific literature standard for SAC (high-fidelity physics / exploration)
LOG_STD_MAX = 2
LOG_STD_MIN = -20  # Expanded support for high-variance exploration
EPSILON = 1e-7


def effnetv2_xs(**kwargs):
    from tmrl.custom.models.EffNetActorCritic import EffNetV2

    """
    Constructs an EfficientNetV2-XS (extra-small) model.
    8 total blocks vs 40 in S — ~5x faster forward pass.
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 16, 1, 1, 0],
        [4, 32, 2, 2, 0],
        [4, 48, 2, 2, 0],
        [4, 96, 3, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_s(**kwargs):
    from tmrl.custom.models.EffNetActorCritic import EffNetV2

    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    from tmrl.custom.models.EffNetActorCritic import EffNetV2

    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    from tmrl.custom.models.EffNetActorCritic import EffNetV2

    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    from tmrl.custom.models.EffNetActorCritic import EffNetV2

    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
