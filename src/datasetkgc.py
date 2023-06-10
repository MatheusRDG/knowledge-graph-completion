from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm


random.seed(42)


class DatasetKGC(Dataset):
    def __init__(self, data):
        self.data = data
        self.data["input_ids"] = self.data["input_ids"]
        self.data["labels"] = self.data["labels"]
        self.data["input_ids"] = self.data["input_ids"]
        self.num_rows = self.data["input_ids"].shape[0]

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        input_ids = self.data["input_ids"][idx].squeeze(0)
        attention_mask = self.data["attention_mask"][idx].squeeze(0)
        labels = self.data["labels"][idx].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def encode_data(data, tokenizer, max_length):
    encoded_input = tokenizer(
        list(data["data_input"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    encoded_label = tokenizer(
        list(data["data_label"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
    )

    examples = []
    for i in tqdm(range(len(data))):
        input_ids = encoded_input["input_ids"][i]
        labels = encoded_label["input_ids"][i]
        attention_mask = encoded_input["attention_mask"][i]
        examples.append(
            {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=max_length
    )

    prepared_data = data_collator(examples)

    return prepared_data


def train_valid_split(data, tokenizer, max_lenght):
    train, valid = train_test_split(data, test_size=0.1, random_state=42)
    return encode_data(train, tokenizer, max_lenght), encode_data(valid, tokenizer, max_lenght)


def generate_train_valid_dataloader(data):
    train, valid = train_valid_split(data)

    train_loader = DataLoader(DatasetKGC(train), batch_size=BATCH_SIZE, shuffle=False)

    valid_loader = DataLoader(DatasetKGC(valid), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader


def generate_train_valid_dataset(data, tokenizer, max_lenght):
    train, valid = train_valid_split(data, tokenizer, max_lenght)

    train_loader = DatasetKGC(train)

    valid_loader = DatasetKGC(valid)

    return train_loader, valid_loader