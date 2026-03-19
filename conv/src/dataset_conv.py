import json
import os
from collections import defaultdict

import torch
import transformers
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import padded_tensor


class CRSConvDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        tokenizer,
        debug=False,
        context_max_length=None,
        resp_max_length=None,
        entity_max_length=None,
        prompt_tokenizer=None,
        prompt_max_length=None,
        n_examples=3,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.debug = debug
        self.n_examples = n_examples
        self.prompt_token = self.prompt_tokenizer.mask_token
        self.context_max_length = context_max_length or self.tokenizer.model_max_length
        self.resp_max_length = (resp_max_length or self.tokenizer.model_max_length) - 1
        self.entity_max_length = entity_max_length or self.tokenizer.model_max_length
        self.prompt_max_length = (prompt_max_length or self.prompt_tokenizer.model_max_length) - 1
        dataset_dir = dataset if isinstance(dataset, str) and os.path.isdir(dataset) else os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        original_log_level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if self.debug:
            lines = lines[:512]

        for line in tqdm(lines):
            dialog = json.loads(line)
            context = '<D>:'
            prompt_context = ''

            for turn_idx, utt in enumerate(dialog['context']):
                if not utt:
                    continue
                speaker = 'User: ' if turn_idx % 2 == 0 else 'System: '
                context += speaker + utt + self.tokenizer.eos_token
                prompt_context += speaker + utt + self.prompt_tokenizer.sep_token

            if context == '<D>:':
                continue

            retrieved_examples = []
            retrieved_gen_examples = []
            for retrieved_context, retrieved_resp in list(zip(dialog['retrieved_contexts'], dialog['un_mask_retrieved_resp']))[: self.n_examples]:
                if retrieved_resp == 'nan':
                    continue
                text_example = ''
                prompt_example = ''
                for demo_turn_idx, utt in enumerate(retrieved_context.split('<s>')):
                    if not utt:
                        continue
                    speaker = 'User: ' if demo_turn_idx % 2 == 0 else 'System: '
                    text_example += speaker + utt + self.tokenizer.eos_token
                    prompt_example += speaker + utt + self.prompt_tokenizer.sep_token
                text_example += f'System: {retrieved_resp}' + self.tokenizer.eos_token
                prompt_example += f'System: {retrieved_resp}' + self.prompt_tokenizer.sep_token
                retrieved_gen_examples.append(text_example)
                retrieved_examples.append(prompt_example)

            retrieved_example_ids = []
            retrieved_gen_example_ids = []
            prompt_prefix = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(self.prompt_token))
            for idx, sent in enumerate(retrieved_examples):
                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(sent))
                prompt_ids = prompt_ids[-self.context_max_length:]
                prompt_ids = prompt_prefix * self.prompt_max_length + prompt_ids
                retrieved_example_ids.append(prompt_ids)

                gen_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(retrieved_gen_examples[idx]))
                gen_ids = gen_ids[-self.context_max_length:]
                retrieved_gen_example_ids.append(gen_ids)

            while len(retrieved_example_ids) < self.n_examples:
                retrieved_example_ids.append(prompt_prefix * (self.prompt_max_length + 1))
                retrieved_gen_example_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<PAD>')))

            context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))[-self.context_max_length:]
            prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))[-self.prompt_max_length:]
            prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

            resp = 'System: ' + dialog['resp']
            resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))[: self.resp_max_length]
            resp_ids.append(self.tokenizer.eos_token_id)

            self.data.append(
                {
                    'context': context_ids,
                    'resp': resp_ids,
                    'entity': list(dialog['retrieved_response_entity'] + dialog['retrieved_context_entity'] + dialog['entity'])[-self.entity_max_length :],
                    'retrieved_example_ids': retrieved_example_ids,
                    'prompt': prompt_ids,
                    'retrieved_example_gen_ids': retrieved_gen_example_ids,
                }
            )
        transformers.logging.set_verbosity(original_log_level)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CRSConvDataCollator:
    def __init__(
        self,
        tokenizer,
        device,
        pad_entity_id,
        gen=False,
        use_amp=False,
        debug=False,
        ignore_pad_token_for_loss=True,
        context_max_length=None,
        resp_max_length=None,
        entity_max_length=None,
        prompt_tokenizer=None,
        prompt_max_length=None,
        n_examples=3,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug
        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None
        self.context_max_length = context_max_length or self.tokenizer.model_max_length
        self.resp_max_length = resp_max_length or self.tokenizer.model_max_length
        self.entity_max_length = entity_max_length or self.tokenizer.model_max_length
        self.prompt_max_length = prompt_max_length or self.prompt_tokenizer.model_max_length
        self.pad_entity_id = pad_entity_id
        self.n_examples = n_examples
        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        retrieved_gen_batch = defaultdict(list)
        entity_batch = []
        resp_batch = []
        context_len_batch = []

        if self.gen:
            self.tokenizer.padding_side = 'left'
            for data in data_batch:
                context_ids = data['context'][-(self.context_max_length - len(self.generate_prompt_ids)) :]
                context_len_batch.append(len(context_ids))
                context_batch['input_ids'].append(context_ids + self.generate_prompt_ids)
                prompt_batch['input_ids'].extend(data['retrieved_example_ids'])
                retrieved_gen_batch['input_ids'].extend(data['retrieved_example_gen_ids'])
                resp_batch.append(data['resp'])
                entity_batch.append(data['entity'])
        else:
            self.tokenizer.padding_side = 'right'
            for data in data_batch:
                context_batch['input_ids'].append((data['context'] + data['resp'])[-self.context_max_length :])
                prompt_batch['input_ids'].extend(data['retrieved_example_ids'])
                retrieved_gen_batch['input_ids'].extend(data['retrieved_example_gen_ids'])
                entity_batch.append(data['entity'])

        input_batch = {}
        context_batch = self.tokenizer.pad(context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, max_length=self.context_max_length)
        if not self.gen:
            resp_batch = [
                [token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp]
                for resp in context_batch['input_ids']
            ]
            input_batch['resp'] = torch.as_tensor(resp_batch, device=self.device)
        else:
            input_batch['resp'] = resp_batch
            input_batch['context_len'] = context_len_batch

        for key, value in context_batch.items():
            if not isinstance(value, torch.Tensor):
                context_batch[key] = torch.as_tensor(value, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(prompt_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, max_length=self.context_max_length)
        for key, value in prompt_batch.items():
            if not isinstance(value, torch.Tensor):
                prompt_batch[key] = torch.as_tensor(value, device=self.device)
        input_batch['prompt'] = prompt_batch

        retrieved_gen_batch = self.tokenizer.pad(retrieved_gen_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of, max_length=self.context_max_length)
        for key, value in retrieved_gen_batch.items():
            if not isinstance(value, torch.Tensor):
                retrieved_gen_batch[key] = torch.as_tensor(value, device=self.device)
        input_batch['retrieved_gen'] = retrieved_gen_batch

        input_batch['entity'] = padded_tensor(
            entity_batch,
            pad_idx=self.pad_entity_id,
            pad_tail=True,
            device=self.device,
            use_amp=self.use_amp,
            debug=self.debug,
            max_len=self.entity_max_length,
        )
        return input_batch
