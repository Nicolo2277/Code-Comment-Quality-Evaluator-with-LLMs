import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help='dataset sample size')
    parser.add_argument('--min_comment_length', type=int, default=3, help='minimun comment length')
    parser.add_argument('--max_code_length', type=int, default=1000, help='maximum code length')
    parser.add_argument('--data_raw', type=str, default='dataset_for_labeling.json', help='dataset that needs to be labeled directory')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (suggested 8/16/32)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter, suggested equals lora_r or 2lora_r')
    parser.add_argument('--lora_dropout', type=int, default=0.1, help='dropout prob in lora')
    parser.add_argument('--learning_rate', type=int, default=2e-5, help='learning rate used in training')
    parser.add_argument('--batch_size_lora', type=int, default=4, help='LoRA batch size training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='reaches efficiently 16 of batch size')
    parser.add_argument('--lora_epochs', type=int, default=10, help='epochs in lora training')
    parser.add_argument('--warmup_ratio_lora', type=int, default=0.1, help='warmup in lora scheduler')
    parser.add_argument('--lora_weight_decay', type=int, default=0.1, help='weight decay in lora training')
    parser.add_argument('--data_file', type=str, required=True, help='json data file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max tokens processed by the model')
    parser.add_argument('--test_size', type=int, default=0.2, help='test dataset size')
    parser.add_argument('--val_size', type=int, default=0.2, help='validation dataset size')
    parser.add_argument('--use_wandb', type=bool, default=False, help='whether or not to log on wandb')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory where files will be saved')

    opt = parser.parse_args()
    return opt

