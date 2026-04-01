import argparse

from easyeditor import (
    FGVEditDataset,
    MultimodalTrainer,
    SERACMultimodalHparams,
    SERACMultimodalTrainingHparams,
)


TRAIN_CONFIGS = {
    "blip2": "hparams/TRAINING/SERAC/blip2.yaml",
    "minigpt4": "hparams/TRAINING/SERAC/minigpt4.yaml",
}

EVAL_CONFIGS = {
    "blip2": "hparams/SERAC/blip2.yaml",
    "minigpt4": "hparams/SERAC/minigpt4.yaml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=TRAIN_CONFIGS.keys(), required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        hparams = SERACMultimodalTrainingHparams.from_hparams(TRAIN_CONFIGS[args.model])
        train_ds = FGVEditDataset("data/FGVEdit/train_data.json", config=hparams)
        eval_ds = FGVEditDataset("data/FGVEdit/test_data.json", config=hparams)
    else:
        hparams = SERACMultimodalHparams.from_hparams(EVAL_CONFIGS[args.model])
        train_ds = FGVEditDataset("data/FGVEdit/test_data.json", config=hparams)
        eval_ds = train_ds

    trainer = MultimodalTrainer(config=hparams, train_set=train_ds, val_set=eval_ds)
    trainer.run()


if __name__ == "__main__":
    main()
