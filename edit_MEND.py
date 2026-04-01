import argparse

from easyeditor import (
    FGVEditDataset,
    MENDMultimodalHparams,
    MENDMultimodalTrainingHparams,
    MultimodalTrainer,
)


TRAIN_CONFIGS = {
    "blip2": "hparams/TRAINING/MEND/blip2.yaml",
    "minigpt4": "hparams/TRAINING/MEND/minigpt4.yaml",
}

EVAL_CONFIGS = {
    "blip2": "hparams/MEND/blip2.yaml",
    "minigpt4": "hparams/MEND/minigpt4.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=TRAIN_CONFIGS.keys(), required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        hparams = MENDMultimodalTrainingHparams.from_hparams(TRAIN_CONFIGS[args.model])
        train_ds = FGVEditDataset("data/FGVEdit/train_data.json", config=hparams)
        eval_ds = FGVEditDataset("data/FGVEdit/test_data.json", config=hparams)
    else:
        hparams = MENDMultimodalHparams.from_hparams(EVAL_CONFIGS[args.model])
        train_ds = FGVEditDataset("data/FGVEdit/test_data.json", config=hparams)
        eval_ds = train_ds

    trainer = MultimodalTrainer(config=hparams, train_set=train_ds, val_set=eval_ds)
    trainer.run()


if __name__ == "__main__":
    main()
