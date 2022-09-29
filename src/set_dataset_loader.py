import glob
from pathlib import Path

import omegaconf

from src.data import JCLDataset
from src.utils import processDataFiles


def set_loader(config):
    metadata_path = Path("../data/raw_datasets/100000")
    cfg = omegaconf.OmegaConf.load("./config.yaml")

    # TODO: load the train dataset
    path = '{}/Train/*.json'.format(config.train_path)
    files = glob.glob(path)
    text = processDataFiles(files)

    text = text.split('\n')  # convert the raw text to a set of examples
    trainText = text[:-1] if len(text[-1]) == 0 else text

    train_dataset = JCLDataset(trainText, metadata_path, cfg, mode="train")

    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                           batch_size=cfg.batch_size,
    #                                           shuffle=True,
    #                                           drop_last=True,
    #                                           # collate_fn=partial(custom_collate_fn, cfg=cfg.dataset_train),
    #                                           num_workers=cfg.num_of_workers,
    #                                           pin_memory=True)

    # TODO：load the val dataset
    path = '{}/Val/*.json'.format(config.val_path)
    files = glob.glob(path)
    textVal = processDataFiles([files[0]])
    textVal = textVal.split('\n')  # convert the raw text to a set of examples
    textVal = textVal[:-1] if len(textVal[-1]) == 0 else textVal

    val_dataset = JCLDataset(textVal, metadata_path, cfg, mode="val")

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                           batch_size=cfg.batch_size,
    #                                           shuffle=False,
    #                                           num_workers=cfg.num_of_workers,
    #                                           # collate_fn=partial(custom_collate_fn, cfg=cfg.dataset_val),
    #                                           pin_memory=True,
    #                                           drop_last=True)

    return train_dataset, val_dataset,  # train_loader, val_loader
