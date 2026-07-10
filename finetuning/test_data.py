from finetuning.data import prepare_dataset

train_dataset, test_dataset, labels = prepare_dataset()

print(train_dataset)

print()

print(train_dataset[0]["text"])