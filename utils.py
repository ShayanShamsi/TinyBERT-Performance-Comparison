TASK_TO_COLUMNS = {
    "sst2": ["sentence"],
    "qnli": ["question", "sentence"],
    "mnli": ["premise", "hypothesis"],
    "qqp": ["question1", "question2"]
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)