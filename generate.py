import torch
# from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d','--dir',dest="dir_of_hf_w", type=str, help='dir folder of hf weights, e.g., xxx.bin')
parser.add_argument('--out-to-txt',dest="out_to_txt", action='store_true', help='store output text to out_generation.txt')

args = parser.parse_args()

tokenizer = LLaMATokenizer.from_pretrained(args.dir_of_hf_w)

model = LLaMAForCausalLM.from_pretrained(
    args.dir_of_hf_w,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
# model = PeftModel.from_pretrained(
#     model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16
# )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""


model.eval()


def evaluate(instruction, input=None, **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    # testing code for readme
    ctx = ""
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        pred = evaluate(instruction)
        ctx += f"Instruction: {instruction}\n" + f"Response: {pred}\n"
        print("Response:", pred)
        print()

    if args.out_to_txt:
        with open("./out_generation.txt",'w') as fp:
            fp.write(ctx)