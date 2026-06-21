# logs/set_logger.py
from loguru import logger
from pathlib import Path
from datetime import datetime


def setup_logger(log_file_path: Path, log_name: str, overwrite: bool):
    """
    Configures the log.

    If overwrite=True, cleans the old file.
    If overwrite=False, it creates a new file with data.
    """

    if overwrite:
        log_file = log_file_path / f"{log_name}.log"
        mode = "w"  # faz com que o arquivo .log de log seja regerado sempre que o código for rodado
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_file_path / f"{log_name}_{timestamp}.log"
        mode = "a"  # faz com que dê 'append' (escreve depois do que já estava la)

    rotation = "10 MB"  # pra evitar que o log fique infinito

    if "model_training" in log_name:
        rotation = "150 MB"  # muito fácil dos arquivos de log de treinamento do modelo ultrapassarem os 100MB

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation=rotation,  # evita que o log fique infinito
        mode=mode,
    )
