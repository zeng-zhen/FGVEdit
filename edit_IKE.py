import argparse
from statistics import mean

from sentence_transformers import SentenceTransformer

from easyeditor import (
    FGVEditDataset,
    IKEMultimodalHyperParams,
    MultimodalEditor,
    encode_ike_facts_multimodal,
)


CONFIG_PATHS = {
    "blip2": "hparams/IKE/blip2.yaml",
    "minigpt4": "hparams/IKE/minigpt4.yaml",
}


def summarize(metrics):
    post = [m["post"] for m in metrics]
    summary = {
        "Reliability": mean(m["rewrite_acc"].item() for m in post),
        "Generality": mean(m["rephrase_acc"].item() for m in post),
        "fg_gen": mean(m["fg_gen_acc"].item() for m in post),
        "fg_loc": mean(m["fg_loc_acc"].item() for m in post),
    }
    summary["Specificity"] = (summary["fg_gen"] + summary["fg_loc"]) / 2
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIG_PATHS.keys(), required=True)
    parser.add_argument("--mode", choices=["embed", "eval"], default="eval")
    args = parser.parse_args()

    hparams = IKEMultimodalHyperParams.from_hparams(CONFIG_PATHS[args.model])
    train_ds = FGVEditDataset("data/FGVEdit/train_data.json", config=hparams)

    sentence_model = SentenceTransformer(hparams.sentence_model_name, cache_folder="./hugging_cache")
    if args.mode == "embed":
        encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
        return

    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
    eval_ds = FGVEditDataset("data/FGVEdit/test_data.json", config=hparams)
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, _, _ = editor.edit_dataset(eval_ds, train_ds=train_ds)
    print(summarize(metrics))


if __name__ == "__main__":
    main()
