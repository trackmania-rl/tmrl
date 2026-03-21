import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA


def test_tm2020_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020Interface(img_hist_len=3, resize_to=None, grayscale=True)
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[3]
    assert img_space.shape == (3, cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH)


def test_impala_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020InterfaceIMPALA(img_hist_len=2, resize_to=None, grayscale=True)
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[-1]
    assert img_space.shape == (2, cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH)
