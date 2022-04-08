import json
import logging
from pathlib import Path

import _jsonnet
import click

from artificial_language.languages import Language
from artificial_language.utils.import_util import import_submodules

logger = logging.getLogger(__name__)
fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=fmt)


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("save-path", type=click.Path(exists=False))
@click.argument("num-sentences", type=int)
@click.option("--batch-size", type=int, default=64)
def generate_artificial_corpus(config_path: str, save_path: str, num_sentences: int, batch_size: int):
    import_submodules("artificial_language")
    config = json.loads(_jsonnet.evaluate_file(config_path))
    language = Language.from_config(config)

    sentence_count = 0
    logger.info(f"Save generated sentences into {save_path}.")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "w") as f:
        while True:
            for sentence in language.batch_generate_sentences(batch_size):
                f.write(" ".join(sentence))
                f.write("\n")
                sentence_count += 1

                if sentence_count == num_sentences:
                    logger.info(f"Generated {sentence_count} sentences.")
                    return


if __name__ == "__main__":
    generate_artificial_corpus()
