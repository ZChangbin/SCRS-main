from typing import List, Optional, Union

import os
import torch
import wandb

PROJECT_NAME = 'SCRS-revised-v1'
RECOMMENDATION = 'recommendation'
GENERATION = 'generation'
MODEL_NAME = 'SCRS'
MODEL_RELATED_PARAMS = ['learning_rate', 'seed']


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False,
) -> torch.LongTensor:
    n = len(items)
    lens = [len(item) for item in items]
    t = max(max(lens), 1)
    if debug and max_len is not None:
        t = max(t, max_len)
    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)
    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item
    return output


def convert_params_to_str(params):
    return ''.join(f'[{key}={value}]' for key, value in params.items() if key in MODEL_RELATED_PARAMS)


def init_wandb_run(project_name, dataset, task, tags, model_name, model_params, type_of_run='full', run_name=None):
    if run_name is None:
        run_name = convert_params_to_str(model_params)
    return wandb.init(
        project=project_name,
        group=f'{dataset}-{task}/',
        job_type=type_of_run,
        tags=tags,
        entity='zcbin-guilin-university-of-electronic-technology',
        reinit=True,
        name=f'{model_name}-{run_name}',
    )


def wandb_logging(eval_dict, step):
    for key, value in eval_dict.items():
        wandb.log(data={key: value}, step=step)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model_params(gen_model, text_encoder, bias_only=True):
    text_encoder.requires_grad_(False)
    if bias_only:
        for param in gen_model.parameters():
            param.requires_grad = False
        for para in gen_model.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)
        for para in text_encoder.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)


def save(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {k: v for k, v in model.state_dict().items() if 'edge' not in k}
    torch.save(state_dict, os.path.join(save_dir, 'model.pt'))


def load(model, load_dir):
    model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pt'), map_location=torch.device('cpu')), strict=False)
    return model


def save_gen_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {
        'pretrained_model': model.pretrained_model.state_dict(),
        'v_head': model.v_head.state_dict(),
    }
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))


def load_gen_model(model, load_dir):
    state_dict = torch.load(os.path.join(load_dir, 'model.pth'), map_location='cpu')
    model.pretrained_model.load_state_dict(state_dict['pretrained_model'])
    model.v_head.load_state_dict(state_dict['v_head'])
    return model


def load_gen_model_new(model, load_dir):
    state_dict = torch.load(os.path.join(load_dir, 'model.pth'), map_location='cpu')
    model.load_state_dict(state_dict['pretrained_model'])
    return model
