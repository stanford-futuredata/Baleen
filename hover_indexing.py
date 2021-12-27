import os
import argparse

from colbert.infra import Run, ColBERTConfig, RunConfig
from colbert import Indexer

from colbert.utils.utils import print_message


def main(args):
    print_message("#> Starting...")

    collection_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.tsv')
    checkpoint_path = os.path.join(args.datadir, 'hover.checkpoints-v1.0/flipr-v1.0.dnn')

    with Run().context(RunConfig(root=args.root)):
        config = ColBERTConfig(doc_maxlen=256, nbits=args.nbits)
        indexer = Indexer(checkpoint_path, config=config)
        indexer.index(name=args.index, collection=collection_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--nbits", type=int, required=True)

    args = parser.parse_args()
    main(args)
