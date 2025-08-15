# utils.py
import torch

def make_prompt(example, prompt_tmpl):
    return prompt_tmpl.format(instruction=example.get('instruction', ''), input=example.get('input', ''))

def get_device_of_model(m):
    try:
        return next(m.parameters()).device
    except StopIteration:
        return torch.device('cpu')