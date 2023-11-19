import argparse
import os

import soft_prompts, prefix, lora, finetune, finetune_last

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Compare performance differences of training methods on a subset of GLUE benchmark using the TinyBERT model.")

    # Add command line arguments
    parser.add_argument('--training_type', type=str, help='One of: public, private.')
    parser.add_argument('--training_method', type=str, help='One of: prompt, prefix, lora, finetune, finetune-last-layer.')
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path of a BERT model hosted on HuggingFace. Ideally, choose a bert-tiny, bert-mini, bert-small, bert-medium, DistilBERT, or BERT model.')
    parser.add_argument('--task', type=str, help='GLUE task to perform. Choose one of: sst2, qnli, mnli, qqp.')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs.')
    parser.add_argument('--lr', type=float, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--p_length', type=int, default=20, help='Token length of soft-prompt or prefix. Defaults to 20.')
    parser.add_argument('--per_sample_max_grad_norm', type=float, default=0.1, help='Clipping gradient')

    # Parse the command line arguments
    args = parser.parse_args()

    if args.training_method == "prompt":
        results, trainable_parameters = soft_prompts.run(args)
    elif args.training_methods == "prefix":
        results, trainable_parameters = prefix.run(args)
    elif args.training_methods == "lora":
        results, trainable_parameters = lora.run(args)
    elif args.training_methods == "finetune":
        results, trainable_parameters = finetune.run(args)
    elif args.training_methods == "finetune_last":
        results, trainable_parameters = finetune_last.run(args)

    # Append information to a file
    with open('output_results.txt', 'a') as file:
        # Append the information to a new line
        file.write(
            f"{args.model_name_or_path}, {args.training_type}, {args.task}, {args.training_method}, {trainable_parameters}, {results}\n"
        )

if __name__ == "__main__":
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
