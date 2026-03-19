import argparse
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from config_copy import deberta_special_tokens_dict, gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv_retrieval_prompt import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator, _compute_distk
from measure_semantic_entropy import Deberta4SE
from model_prompt import KGPrompt
from utils import (
    GENERATION,
    MODEL_NAME,
    PROJECT_NAME,
    count_parameters,
    freeze_model_params,
    init_wandb_run,
    load,
    load_gen_model,
    save,
    save_gen_model,
    wandb_logging,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--context_max_length", type=int, default=200, help="max length of both encoder and decoder input.")
    parser.add_argument("--resp_max_length", type=int, default=80, help="max length of decoder input.")
    parser.add_argument("--entity_max_length", type=int, default=64, help="max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="/data/zhongchangbin/SCRS-base_model/DialoGPT-small")
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true", default=True)
    parser.add_argument("--text_tokenizer", type=str, default="/data/zhongchangbin/SCRS-base_model/roberta-base")
    parser.add_argument("--model", type=str, default="/data/zhongchangbin/SCRS-base_model/DialoGPT-small")
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str, default="/data/zhongchangbin/SCRS-base_model/roberta-base")
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int, default=110)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    parser.add_argument("--n_examples", type=int, default=3, help="number of retrieved demonstrations")
    parser.add_argument("--quantile_threshold", type=float, default=0.5, help="the threshold of quantile")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--fp16", action="store_true", help="use automatic mixed precision to speed up.")
    parser.add_argument("--mapping", action="store_true", default=True, help="if we use semantic mapping")
    parser.add_argument("--bias_only", action="store_true", help="if we use semantic mapping")
    parser.add_argument("--ppo_config_profile",type=str,default="redial",choices=["auto", "inspired", "redial"],help="which PPOConfig profile to use")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    parser.add_argument("--type_of_run", default="full", help="type of the experiment, eg: full, ablation, analysis")
    parser.add_argument("--check_point", type=str,default="./checkpoint/trained_redial-seed42/best", help="path to the check point")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        args.use_wandb = False

    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if accelerator.is_local_main_process else "ERROR")
    logger.add(f"log/{local_time}.log", level="DEBUG" if accelerator.is_local_main_process else "ERROR")
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.use_wandb:
        name = args.name if args.name else local_time
        name += "_" + str(accelerator.process_index)
        if args.log_all:
            group = args.name if args.name else "DDP_" + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model)
    ref_model.pretrained_model.resize_token_embeddings(len(tokenizer))
    ref_model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    se_tokenizer = AutoTokenizer.from_pretrained("/data/zhongchangbin/SCRS-base_model/deberta-v2-xlarge-mnli")
    se_tokenizer.add_special_tokens(deberta_special_tokens_dict)
    se_model = AutoModelForSequenceClassification.from_pretrained(
        "/data/zhongchangbin/SCRS-base_model/deberta-v2-xlarge-mnli"
    )
    se_model.resize_token_embeddings(len(se_tokenizer))
    se_model = se_model.to(device)

    print(tokenizer.pad_token_id)
    print(tokenizer.encode("<movie>"))

    prompt_encoder = KGPrompt(
        model.config.n_embd,
        text_encoder.config.hidden_size,
        model.config.n_head,
        model.config.n_layer,
        2,
        n_entity=kg["num_entities"],
        num_relations=kg["num_relations"],
        num_bases=args.num_bases,
        edge_index=kg["edge_index"],
        edge_type=kg["edge_type"],
        n_prefix_rec=args.n_prefix_conv,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples,
    )

    if args.check_point is not None:
        prompt_encoder = load(prompt_encoder, args.check_point + "/prompt_encoder")
        model = load_gen_model(model, args.check_point + "/gen_model")
        ref_model = load_gen_model(ref_model, args.check_point + "/gen_model")

    init_wandb_run(
        project_name=PROJECT_NAME,
        dataset=args.dataset,
        task=GENERATION,
        model_name=MODEL_NAME,
        model_params=vars(args),
        type_of_run=args.type_of_run,
        tags="Method",
        run_name=None,
    )

    prompt_encoder = prompt_encoder.to(device)
    freeze_model_params(model, text_encoder, bias_only=args.bias_only)

    print("Total numbef of trainable gen params: ", count_parameters(model))
    print("Total numbef of trainable prompt params: ", count_parameters(text_encoder))

    modules = [model]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for m in modules
                for n, p in m.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for m in modules
                for n, p in m.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = CRSConvDataset(
        args.dataset,
        "train",
        tokenizer,
        debug=args.debug,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples,
    )
    valid_dataset = CRSConvDataset(
        args.dataset,
        "valid",
        tokenizer,
        debug=args.debug,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples,
    )
    test_dataset = CRSConvDataset(
        args.dataset,
        "test",
        tokenizer,
        debug=args.debug,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer,
        prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples,
    )
    data_collator_teacher = CRSConvDataCollator(
        tokenizer=tokenizer,
        device=device,
        use_amp=accelerator.use_fp16,
        debug=args.debug,
        gen=False,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length + args.resp_max_length,
        entity_max_length=args.entity_max_length,
        pad_entity_id=kg["pad_entity_id"],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer,
        device=device,
        gen=True,
        use_amp=accelerator.use_fp16,
        debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length,
        resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        pad_entity_id=kg["pad_entity_id"],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length,
    )
    train_gen_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
        shuffle=True,
    )
    valid_gen_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    test_gen_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )

    gen_file_path = os.path.join("log", f"gen_{local_time}.jsonl")
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)

    model, prompt_encoder, train_gen_dataloader, se_model = accelerator.prepare(
        model, prompt_encoder, train_gen_dataloader, se_model
    )

    deberta4SE = Deberta4SE(se_tokenizer, se_model)

    num_update_steps_per_epoch = math.ceil(len(train_gen_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric, mode = "dist@4", 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float("inf")
    best_metric_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_metric_dir, exist_ok=True)

    if args.ppo_config_profile == "auto":
        dataset_lower = str(args.dataset).lower()
        if "redial" in dataset_lower:
            ppo_profile = "redial"
        elif "inspired" in dataset_lower:
            ppo_profile = "inspired"
        else:
            ppo_profile = "inspired"
    else:
        ppo_profile = args.ppo_config_profile

    if ppo_profile == "redial":
        ppo_config = PPOConfig(
            batch_size=args.per_device_train_batch_size,
            mini_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ratio_threshold=5.0,
            max_grad_norm=0.5,
            ppo_epochs=2,
            learning_rate=args.learning_rate,
            early_stopping=True,
            whiten_rewards=True,
            init_kl_coef=0.4,
            log_with="wandb",
        )
    else:
        ppo_config = PPOConfig(
            batch_size=args.per_device_train_batch_size,
            mini_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ratio_threshold=5.0,
            max_grad_norm=0.5,
            ppo_epochs=3,
            learning_rate=args.learning_rate,
            early_stopping=True,
            whiten_rewards=True,
            init_kl_coef=0.3,
            log_with="wandb",
        )

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=ppo_config,
        dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_generator,
        optimizer=optimizer,
    )

    ppo_trainer = accelerator.prepare(ppo_trainer)

    special_tokens_to_remove = [re.escape(tok) for tok in tokenizer.all_special_tokens if tok != "<movie>"]
    pattern = "|".join(special_tokens_to_remove)

    for epoch in range(args.num_train_epochs):
        model.train()
        semantic_entropy_history = []
        valid_count = 0
        ppo_buffer = {
            "queries": [],
            "responses": [],
            "scores": [],
            "response_masks": [],
        }
        for step, gen_batch in enumerate(train_gen_dataloader):
            with torch.no_grad():
                gen_token_embeds = text_encoder(**gen_batch["prompt"]).last_hidden_state

                gen_prompt_augmented_input_embeddings, gen_new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=gen_batch["entity"],
                    token_embeds=gen_token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping=args.mapping,
                    word_embeddings=model.pretrained_model.get_input_embeddings().weight,
                    context_input_embeddings=model.pretrained_model.get_input_embeddings()(
                        gen_batch["context"]["input_ids"]
                    ),
                    attention_mask=gen_batch["context"]["attention_mask"],
                )

                gen_batch["context"]["inputs_embeds"] = gen_prompt_augmented_input_embeddings
                gen_batch["context"]["attention_mask"] = gen_new_attention_mask

            generated_sequences = []
            log_likelihoods = []
            token_entropies = []
            mutual_informations = []
            num_generation = 5
            batch_size = gen_batch["context"]["input_ids"].size(0)
            dist2 = []

            for i in range(3):
                temperature_i = 0.1 if i == 0 else 0.5
                num_i = 1 if i == 0 else 2
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(ppo_trainer.model).generate(
                        inputs_embeds=gen_batch["context"]["inputs_embeds"],
                        attention_mask=gen_batch["context"]["attention_mask"],
                        max_new_tokens=50,
                        output_scores=True,
                        output_hidden_states=False,
                        no_repeat_ngram_size=3,
                        return_dict_in_generate=True,
                        temperature=temperature_i,
                        do_sample=True,
                        top_p=0.7,
                        top_k=30,
                        num_return_sequences=num_i,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_seqs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
                gen_seqs = [re.sub(pattern, "", seq) for seq in gen_seqs]
                generated_sequences.append(np.array(gen_seqs).reshape(-1, num_i))

                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id
                token_entropy = _token_entropy_from_scores(
                    outputs.scores,
                    sequences=outputs.sequences,
                    batch_size=batch_size,
                    num_return_sequences=num_i,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                )
                token_entropies.append(token_entropy)
                if num_i > 1:
                    mi = _mutual_information_from_scores(
                        outputs.scores,
                        sequences=outputs.sequences,
                        batch_size=batch_size,
                        num_return_sequences=num_i,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                    )
                    mutual_informations.append(mi)

                transition_scores = model.pretrained_model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True,
                )
                transition_scores = torch.where(transition_scores == -float("inf"), torch.nan, transition_scores)
                transition_scores = transition_scores.view(-1, num_i, transition_scores.shape[-1])
                log_likelihoods.append(torch.nanmean(transition_scores, dim=-1))

                dis2 = _compute_distk(tokenizer, outputs.sequences, num_i=num_i, k=2)
                dist2.append(dis2)
                del transition_scores

            generated_sequences = np.concatenate(generated_sequences, axis=1)
            log_likelihood = torch.cat(log_likelihoods, dim=1).requires_grad_(True)

            sequence_based_entropy = _sequence_based_entropy_from_log_likelihood(log_likelihood.detach().cpu())

            token_entropy_metric = torch.cat(token_entropies, dim=1).mean(dim=1).detach().cpu()
            if len(mutual_informations) > 0:
                mutual_information_metric = torch.stack(mutual_informations, dim=0).mean(dim=0).detach().cpu()
            else:
                mutual_information_metric = torch.zeros(batch_size).cpu()

            dist2 = np.concatenate(dist2, axis=1)
            dist2_mean = np.mean(dist2, axis=1)

            if step == len(train_gen_dataloader) - 1:
                print(generated_sequences)

            relations = deberta4SE.get_relations(generated_sequences)
            se_id = deberta4SE.get_se_id(relations, num_generation)
            se_id = torch.tensor(se_id, device=device)
            semantic_entropy = deberta4SE.get_semantic_entropy(se_id, log_likelihood).cpu()
            reward = torch.tensor(dist2_mean) - semantic_entropy
            queries_tensor = gen_batch["context"]["input_ids"]
            system_length = 2
            response_tensors = [torch.tensor(resp[system_length:]) for resp in gen_batch["resp"]]
            response_tensors = pad_sequence(
                response_tensors,
                batch_first=True,
                padding_value=tokenizer.eos_token_id,
            )
            response_mask = (response_tensors != tokenizer.eos_token_id).float()

            semantic_entropy_history.extend(uncertainty.numpy().tolist())
            threshold = np.quantile(semantic_entropy_history, args.quantile_threshold)
            valid_indices = [i for i, se in enumerate(uncertainty) if se >= threshold]
            valid_count += len(valid_indices)

            if valid_indices:
                selected_queries = queries_tensor[valid_indices]
                selected_responses = response_tensors[valid_indices]
                selected_rewards = reward[valid_indices]
                selected_masks = response_mask[valid_indices]

                ppo_buffer["queries"].extend(selected_queries.unbind(0))
                ppo_buffer["responses"].extend(selected_responses.unbind(0))
                ppo_buffer["scores"].extend(selected_rewards.unbind(0))
                ppo_buffer["response_masks"].extend(selected_masks.unbind(0))

                while len(ppo_buffer["queries"]) >= ppo_config.batch_size:
                    batch_data = {
                        "queries": ppo_buffer["queries"][: ppo_config.batch_size],
                        "responses": ppo_buffer["responses"][: ppo_config.batch_size],
                        "scores": ppo_buffer["scores"][: ppo_config.batch_size],
                        "response_masks": ppo_buffer["response_masks"][: ppo_config.batch_size],
                    }

                    ppo_trainer.step(**batch_data)
                    ppo_buffer["queries"] = ppo_buffer["queries"][ppo_config.batch_size :]
                    ppo_buffer["responses"] = ppo_buffer["responses"][ppo_config.batch_size :]
                    ppo_buffer["scores"] = ppo_buffer["scores"][ppo_config.batch_size :]
                    ppo_buffer["response_masks"] = ppo_buffer["response_masks"][ppo_config.batch_size :]

            progress_bar.update(1)
            del gen_batch

        logger.info(f"valid count: {valid_count}")

        valid_loss = []
        model.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch["prompt"]).last_hidden_state
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch["entity"],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping=args.mapping,
                    word_embeddings=model.pretrained_model.get_input_embeddings().weight,
                    context_input_embeddings=model.pretrained_model.get_input_embeddings()(
                        batch["context"]["input_ids"]
                    ),
                    attention_mask=batch["context"]["attention_mask"],
                )
                batch["context"]["inputs_embeds"] = prompt_augmented_input_embeddings
                batch["context"]["attention_mask"] = new_attention_mask

                pad_resp = -100 * torch.ones(
                    (new_attention_mask.shape[0], args.n_examples * args.prompt_max_length)
                ).to(new_attention_mask.device).long()
                batch["resp"] = torch.cat([pad_resp, batch["resp"]], dim=1)

                loss = model.pretrained_model(
                    inputs_embeds=batch["context"]["inputs_embeds"],
                    input_ids=None,
                    labels=batch["resp"],
                    return_dict=True,
                )["loss"]
                valid_loss.append(float(loss))

        evaluator.log_file.write(f"\n\n*** valid-{evaluator.log_cnt} ***\n\n")
        for batch in tqdm(valid_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch["prompt"]).last_hidden_state
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch["entity"],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping=args.mapping,
                    word_embeddings=model.pretrained_model.get_input_embeddings().weight,
                    context_input_embeddings=model.pretrained_model.get_input_embeddings()(
                        batch["context"]["input_ids"]
                    ),
                    attention_mask=batch["context"]["attention_mask"],
                )
                batch["context"]["inputs_embeds"] = prompt_augmented_input_embeddings
                batch["context"]["attention_mask"] = new_attention_mask
                gen_seqs = accelerator.unwrap_model(model.pretrained_model).generate(
                    input_ids=None,
                    inputs_embeds=batch["context"]["inputs_embeds"],
                    attention_mask=batch["context"]["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch["context_len"]):
                    gen_seq = [token_id.item() for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)
                evaluator.evaluate(gen_resp_ids, batch["resp"], log=accelerator.is_local_main_process)

        accelerator.wait_for_everyone()
        report = evaluator.report()
        valid_report = {}
        for k, v in report.items():
            valid_report[f"valid/{k}"] = v
        valid_loss = np.mean(valid_loss)
        valid_report["valid/loss"] = valid_loss
        valid_report["epoch"] = epoch
        wandb_logging(eval_dict=valid_report, step=epoch)
        logger.info(valid_report)
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f"valid/{metric}"] * mode > best_metric * mode:
            best_metric = valid_report[f"valid/{metric}"]
            save(prompt_encoder, f"{best_metric_dir}/prompt_encoder")
            save_gen_model(model, f"{best_metric_dir}/gen_model")
            logger.info(f"new best model with {metric}")

        test_loss = []
        model.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch["prompt"]).last_hidden_state
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch["entity"],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping=args.mapping,
                    word_embeddings=model.pretrained_model.get_input_embeddings().weight,
                    context_input_embeddings=model.pretrained_model.get_input_embeddings()(
                        batch["context"]["input_ids"]
                    ),
                    attention_mask=batch["context"]["attention_mask"],
                )
                batch["context"]["input_ids"] = None
                batch["context"]["inputs_embeds"] = prompt_augmented_input_embeddings
                batch["context"]["attention_mask"] = new_attention_mask

                pad_resp = -100 * torch.ones(
                    (new_attention_mask.shape[0], args.n_examples * args.prompt_max_length)
                ).to(new_attention_mask.device).long()
                batch["resp"] = torch.cat([pad_resp, batch["resp"]], dim=1)

                loss = model.pretrained_model(
                    inputs_embeds=batch["context"]["inputs_embeds"],
                    input_ids=None,
                    labels=batch["resp"],
                    return_dict=True,
                )["loss"]
                test_loss.append(float(loss))

        evaluator.log_file.write(f"\n*** test-{evaluator.log_cnt} ***\n\n")
        for batch in tqdm(test_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch["prompt"]).last_hidden_state
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch["entity"],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping=args.mapping,
                    word_embeddings=model.pretrained_model.get_input_embeddings().weight,
                    context_input_embeddings=model.pretrained_model.get_input_embeddings()(
                        batch["context"]["input_ids"]
                    ),
                    attention_mask=batch["context"]["attention_mask"],
                )
                batch["context"]["input_ids"] = None
                batch["context"]["inputs_embeds"] = prompt_augmented_input_embeddings
                batch["context"]["attention_mask"] = new_attention_mask
                gen_seqs = accelerator.unwrap_model(model.pretrained_model).generate(
                    input_ids=None,
                    inputs_embeds=batch["context"]["inputs_embeds"],
                    attention_mask=batch["context"]["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch["context_len"]):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)
                evaluator.evaluate(gen_resp_ids, batch["resp"], log=accelerator.is_local_main_process)

        accelerator.wait_for_everyone()
        report = evaluator.report()
        test_report = {}
        for k, v in report.items():
            test_report[f"test/{k}"] = v
        test_loss = np.mean(test_loss)
        test_report["test/loss"] = test_loss
        test_report["epoch"] = epoch
        wandb_logging(eval_dict=test_report, step=epoch)
        logger.info(test_report)
        if run:
            run.log(test_report)
        evaluator.reset_metric()

        evaluator.log_cnt += 1

    final_dir = os.path.join(args.output_dir, "final")
    save(prompt_encoder, f"{final_dir}/prompt_encoder")
    save_gen_model(model, f"{final_dir}/gen_model")
    logger.info("save final model")

    wandb.finish()
