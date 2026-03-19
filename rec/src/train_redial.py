import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from item_information import item_comment, SemanticMapping, RankingLoss_new
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec_copy import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
import torch.nn.functional as F
from efficiency_monitor import EfficiencyMonitor



    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='./checkpoint/train-seed42-redial', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--entity_max_length", type=int, default=32, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default="/data/zhongchangbin/SCRS-base_model/DialoGPT-small")
    parser.add_argument("--text_tokenizer", type=str, default="/data/zhongchangbin/SCRS-base_model/roberta-base")
    # model
    parser.add_argument("--model", type=str, default="/data/zhongchangbin/SCRS-base_model/DialoGPT-small",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, default="/data/zhongchangbin/SCRS-base_model/roberta-base")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int, default=10)
    parser.add_argument("--prompt_encoder", type=str, default="./checkpoint/pretrain-seed42-redial/best")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=530)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
 
    parser.add_argument("--k_value", type=int, default=50)
    parser.add_argument("--lambda_value", type=float, default=0.7)

    parser.add_argument(
        "--item_embed_fusion",
        type=str,
        default="proposed",
        choices=["cross_attn", "gate", "proposed", "kg_only", "text_only", "kgsf_mi", "add"],
    )

    parser.add_argument("--mi_align_weight", type=float, default=0.0)
    parser.add_argument("--mi_align_temperature", type=float, default=0.07)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    if args.output_dir is not None:
        out = os.path.normpath(args.output_dir)
        if not os.path.isabs(out) and not out.startswith('checkpoint'):
            args.output_dir = os.path.join('checkpoint', out.lstrip('./'))

    if args.item_embed_fusion == 'kgsf_mi' and args.mi_align_weight == 0.0:
        args.mi_align_weight = 0.01

    def _info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = (a @ b.t()) / temperature
        labels = torch.arange(a.size(0), device=a.device)
        return F.cross_entropy(logits, labels)

    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # 2025.1.17 item_comment和item_embeding
    item_comment = item_comment(dataset = 'redial', model_name_or_path=(args.text_encoder or args.text_tokenizer))
    item_embeds = item_comment.get_item_embedding().to(device)
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    # model_gpt2.py
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # data
    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    
    # 初始化效率监控器
    efficiency_monitor = EfficiencyMonitor(output_dir=args.output_dir, device=str(device))
    logger.info("Efficiency monitoring enabled")
    
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num valid examples = {len(valid_dataset)}")

    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    
    lamda = 0.9
    prev_val_loss = float('inf') 
    
    # train loop
    efficiency_monitor.start_total_training()
    
    for epoch in range(args.num_train_epochs):
        efficiency_monitor.start_epoch_training()
        train_loss = []
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds  
            batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            # SCRS
            outputs = model(**batch['context'], rec=True)
            loss = outputs.rec_loss / args.gradient_accumulation_steps

            if (not efficiency_monitor.flops_computed) and accelerator.is_local_main_process:
                try:
                    class _FlopsWrapper(torch.nn.Module):
                        def __init__(self, m):
                            super().__init__()
                            self.m = m

                        def forward(self, context):
                            return self.m(**context, rec=True)

                    flops_model = _FlopsWrapper(accelerator.unwrap_model(model))
                    sample_context = {k: v for k, v in batch['context'].items() if isinstance(v, torch.Tensor)}
                    efficiency_monitor.compute_flops(flops_model, (sample_context,), mode='train')
                except Exception:
                    pass
            if args.lambda_value == 0.0:
                total_loss = loss
            else:
                item_embeds_adust = prompt_encoder.get_adust_item_embeds(
                    batch['context']['entity_embeds'][kg['item_ids'], :],
                    item_embeds,
                    fusion_mode=args.item_embed_fusion,
                )
                id_to_index = {item_id: idx for idx, item_id in enumerate(kg['item_ids'])}
                rec_label_indices = torch.tensor([id_to_index[label] for label in batch['context']['rec_labels'].tolist()], dtype=torch.long)
                #  获取真实标签对应的 embedding
                ground_truth_embeds_adust = item_embeds_adust[rec_label_indices,:]
                item_logits = outputs.rec_logits[:, kg['item_ids']] 
                cosine_sim = F.cosine_similarity(ground_truth_embeds_adust.unsqueeze(1), item_embeds_adust.unsqueeze(0), dim=-1).detach() 
                RankLoss_new = item_comment.RankingLoss_new(cosine_sim, item_logits,k_value = args.k_value)   
                total_loss =  loss + args.lambda_value* RankLoss_new

            if args.mi_align_weight > 0.0:
                id_to_index = {item_id: idx for idx, item_id in enumerate(kg['item_ids'])}
                rec_label_indices = torch.tensor([id_to_index[label] for label in batch['context']['rec_labels'].tolist()], dtype=torch.long)
                view_entity = batch['context']['entity_embeds'][kg['item_ids'], :][rec_label_indices, :]
                view_text = item_embeds[rec_label_indices, :]
                mi_align_loss = _info_nce(view_entity, view_text, args.mi_align_temperature)
                total_loss = total_loss + args.mi_align_weight * mi_align_loss

            if args.item_embed_fusion == 'kgsf_mi':
                id_to_index = {item_id: idx for idx, item_id in enumerate(kg['item_ids'])}
                rec_label_indices = torch.tensor([id_to_index[label] for label in batch['context']['rec_labels'].tolist()], dtype=torch.long)
                nodes_kg = batch['context']['entity_embeds'][kg['item_ids'], :]
                nodes_text = item_embeds
                user_text = nodes_text[rec_label_indices, :]
                user_kg = nodes_kg[rec_label_indices, :]
                scores_kg = user_text @ nodes_kg.t()
                scores_text = user_kg @ nodes_text.t()
                labels = F.one_hot(rec_label_indices, num_classes=nodes_kg.size(0)).float().to(scores_kg.device)
                info_db_loss = F.mse_loss(scores_kg, labels, reduction='none').sum(dim=-1).mean()
                info_con_loss = F.mse_loss(scores_text, labels, reduction='none').sum(dim=-1).mean()
                total_loss = total_loss + 0.025 * (info_db_loss + info_con_loss)
            
            accelerator.backward(total_loss)
            train_loss.append(float(total_loss))
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        epoch_train_time = efficiency_monitor.end_epoch_training()
        logger.info(f'epoch {epoch} train loss {train_loss}, training time: {epoch_train_time:.1f}s')

        del train_loss, batch

        # valid
        valid_loss = []
        prompt_encoder.eval()
        efficiency_monitor.start_epoch_eval()
        for batch in tqdm(valid_dataloader):
            batch_size = batch['context']['input_ids'].size(0)
            start_time = efficiency_monitor.measure_batch_latency(batch_size)
            
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)
            
            efficiency_monitor.end_batch_latency(start_time, batch_size)
        
        epoch_eval_time = efficiency_monitor.end_epoch_eval()
        
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(valid_report)
        logger.info(f'Epoch {epoch} eval time: {epoch_eval_time:.1f}s')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        # 2024.12.18
        # lamda = adjust_lambda(initial_lambda=lamda, valid_loss=np.mean(valid_loss), prev_valid_loss=prev_val_loss)
        # prev_val_loss = valid_loss
        
        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')
        
        # test
        test_loss = []
        prompt_encoder.eval()

        ## case study
        # import json
        # matched_results = []  # 存储所有匹配的样本信息
        # index_i = 0  # 追踪全局样本编号
        
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                evaluator.evaluate(ranks, labels)


        #         # case study
        #         batch_size = len(labels)  # 当前 batch 样本数
        #         for i in range(len(labels)):
        #             label = labels[i].item()  # 获取第 i 个 label 的值（转为 Python int）
        #             top1 = ranks[i][0]        # 获取第 i 个样本的 top-1 推荐项
        #             entity = batch['entity'][i].tolist()   # 当前样本对应的实体 ID
                    
        #             global_index = index_i + i  # 全局样本编号

        #             if label == top1:
        #                 print(f"Match found at index {global_index}: label = {label}, top1 = {top1}")
        #                 print(f"entity:{entity}")
        #                 matched_sample = {
        #                         'global_index': global_index,
        #                         'label': label,
        #                         'top1': top1,
        #                         'entity': entity
        #                 }
        #                 matched_results.append(matched_sample)
        #         index_i += batch_size  # 累加样本总数
        # # 写入到文件（每行一个JSON对象）
        # with open('matched_samples_redial.jsonl', 'w', encoding='utf-8') as f:
        #     for sample in matched_results:
        #         f.write(json.dumps(sample) + '\n')
        # print(f"{len(matched_results)} matched samples saved to 'matched_samples.jsonl'")

        
        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
    
    # 结束总训练计时并保存效率统计
    efficiency_monitor.end_total_training()
    efficiency_monitor.print_summary()
    efficiency_monitor.save_to_file('efficiency_stats.json')
    if run:
        efficiency_monitor.save_to_wandb(run, prefix='efficiency')
