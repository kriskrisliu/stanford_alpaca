import torch
# from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
import argparse
import json
from tqdm import tqdm

def evaluate(instruction, tokenizer, model, input=None, **kwargs):
    args = kwargs.get("args")
    prompt = generate_prompt(instruction, args, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda() # these are integers encoded from words
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
    output = tokenizer.decode(s) # this will return a fully-wholely description like "Below is an instruction....Response:..."
    return output.split("### Output:")[1].strip()


def generate_prompt(instruction, args, input=None):
    if args.prompt_mode=="continue":
        return f"""Below is a text extending task. you will be given an incomplete text and requested to provide a continuation of said text in the !!!LAION-6plus-style!!!.
### Input:
{instruction}
### Output:"""
    elif args.prompt_mode=="blip_pair":
        return f"""Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.
### Input:
{instruction}
### Output:"""


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d','--dir',dest="dir_of_hf_w", type=str, help='dir folder of hf weights, e.g., xxx.bin')
    parser.add_argument('--out-to-txt',dest="out_to_txt", action='store_true', help='store output text to out_generation.txt')
    parser.add_argument('--load-in-8bit',dest="load_in_8bit", action='store_true', help='')
    parser.add_argument('-i','--interact',dest="interact", action='store_true', help='')
    parser.add_argument("--coco",'--coco_caption',dest="coco_caption", action='store_true', help='generate refined captions for coco, save as json')
    parser.add_argument('--prompt-mode',dest="prompt_mode",type=str, help='[blip_pair, continue]')

    args = parser.parse_args()

    # building the model and tokenizer
    tokenizer = LLaMATokenizer.from_pretrained(args.dir_of_hf_w)
    model = LLaMAForCausalLM.from_pretrained(
        args.dir_of_hf_w,
        load_in_8bit=args.load_in_8bit, # by Kris: True may save memory (16GB to 10GB), but slower
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # model = PeftModel.from_pretrained(
    #     model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16
    # )
    model.eval()
    
    if args.interact:
        print("For testing, please input your prompt:\n")
        instruction_from_terminal = " "
        while instruction_from_terminal!="exit":
            instruction_from_terminal = input("Your prompt: ")
            pred = evaluate(instruction_from_terminal,tokenizer, model,args=args)
            print("Response:", pred)
            print()
        # if type "exit" in terminal, will go on for some examples.
    elif args.coco_caption:
        new_caption = []
        with open("./coco_captions_val.json","r") as fp:
            coco_json = json.load(fp)
        for it in tqdm(coco_json):
            cap =  it['caption']
            pred = evaluate(cap, tokenizer, model,args=args)
            pred = pred.replace('</s>','')
            it["refined"] = pred
            new_caption += [it]
        with open(f"./coco_captions_val_pairs_{args.prompt_mode}.json","w") as fp:
            json.dump(new_caption,fp,indent=4)
    else:
        ctx = ""
        for instruction in [
            'A big bus in a parking lot next to a big building ',
            'There is a clock right outside of the tall building.',
            'Two trains on the track at a railway.',
            'The bench is in a shady area surrounded by plants',
            'an image of man having lunch with kids',
            'A group of people sitting at a table with food.',
            'Flowers neatly arranged in a clear vase filled with water. ',
            'Two men playing frisbee together on a field',
            'A baseball game where the batter is waiting for the pitch.',
            'A motorcycle is parked inside of a building.'
        ]:
            print("Instruction:", instruction)
            pred = evaluate(instruction, tokenizer, model,args=args)
            ctx += f"Instruction: {instruction}\n" + f"Response: {pred}\n"
            print("Response:", pred)
            print()

        if args.out_to_txt:
            with open("./out_generation.txt",'w') as fp:
                fp.write(ctx)

if __name__ == "__main__":
    # testing code for readme
    main()