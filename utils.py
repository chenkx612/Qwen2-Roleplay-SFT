# utils.py

def total_length(sample):
    return (
        len(sample['instruction']) + 
        len(sample['input']) + 
        len(sample['output'])
    )

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
    print(f'可训练参数: {trainable} / {total} ({100*trainable/total:.2f}%)')

def find_sublist(hay, needle):
    if not needle:
        return -1
    Lh = len(hay)
    Ln = len(needle)
    for i in range(Lh - Ln, -1, -1):
        if hay[i:i+Ln] == needle:
            return i
    return -1
