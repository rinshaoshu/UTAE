BANDS_CONFIG = [
    {'dir': 's2', 'bands': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                            'B08', 'B8A', 'B11', 'B12', 'DEM']},
    {'dir': 'asc', 'bands': ['AVV', 'AVH','DEM']},
    {'dir': 'dsc', 'bands': ['DVH', 'DVV','DEM']},
]


# noinspection PyUnhashable
def get_bands_by_dirs(dir_list):
    """根据dir列表按顺序返回对应的配置"""
    dir_to_config = {item['dir']: item for item in BANDS_CONFIG}

    try:
        return [dir_to_config[dir_name] for dir_name in dir_list]
    except KeyError as e:
        available = list(dir_to_config.keys())
        raise ValueError(f"未知的dir: '{e.args[0]}'，可用选项: {available}")

