import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
model_type = 'gpt2'
device = 'cpu'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval()


def generate(prompt='', num_samples=10, steps=20, do_sample=True):
        
    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = tokenizer(prompt).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '': 
            # to create unconditional samples...
            # huggingface/transformers tokenizer special cases these strings
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
        x = encoded_input['input_ids']
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)


        
def generate_patched_activations(prompt1, prompt2, words_to_compare, num_samples=1, steps=1, do_sample=False):
    tokenizer = BPETokenizer()
    
    prompt1_tokenized = tokenizer(prompt1).to(device)
    prompt2_tokenized = tokenizer(prompt2).to(device)

    assert prompt1_tokenized.shape == prompt2_tokenized.shape, "Prompt 1 and Prompt 2 should have the same shape"

    x1 = prompt1_tokenized.expand(num_samples, -1)
    x2 = prompt2_tokenized.expand(num_samples, -1)

    print("Prompt 1: ", prompt1)
    print("Prompt 2: ", prompt2)
    print(f"Prompt 1 tokenized: {x1}")
    print(f"Prompt 2 tokenized: {x2}")

    # forward the model `steps` times to get samples, in a batch
    y, probs_next, tokens_idx = model.generate_mod(x1, max_new_tokens=steps, do_sample=do_sample, save_activations=True, patch_activations=False)
    
    next_tokens = [tokenizer.decode(tokens_idx.t()[i, :]) for i in range(tokens_idx.size(1))]
    probs_next = probs_next.squeeze().numpy()    
    tokens_idx = tokens_idx.squeeze().numpy()
    
    create_table(probs_next, tokens_idx, next_tokens)# Model outputs

    y = model.generate_mod(x2, max_new_tokens=steps, do_sample=do_sample, save_activations=False, patch_activations=True)

    # for key, value in model.outputs.items():
    #     print(f"{key}: {value.shape}")

    print(f"Words to compare: {words_to_compare}  {tokenizer(words_to_compare[0])} {tokenizer(words_to_compare[1])}")

    matrix = torch.zeros((12, model.len_saved_tokens))
    for i in range(12):
        for j in range(model.len_saved_tokens):
            matrix[i, j] = model.outputs[f'layer_{i}_pos_{j}'][:, tokenizer(words_to_compare[0])] - model.outputs[f'layer_{i}_pos_{j}'][:, tokenizer(words_to_compare[1])]
    
    prompt_1_decoded = [tokenizer.decode(prompt1_tokenized.t()[i, :]) for i in range(prompt1_tokenized.size(1))]
    
    plot_matrix(matrix, prompt_1_decoded, 12)

    
    


def create_table(probs_next, tokens_idx, next_tokens):
    df = pd.DataFrame({'Probability': probs_next, 'Token index': tokens_idx, 'Token': next_tokens})
    # Hacer que la columna index se llame "Position" y que sea la primera columna
    df.index.name = 'Position'
    print(tabulate(df, headers='keys', tablefmt="simple"))#'psql'

def plot_matrix(matrix, token_list:list, n_layers:int = 12, save = True):
    """
    Plots the activation matrix.

    Args:
        matrix (numpy.ndarray): The activation matrix.
        token_list (list): The list of tokens.
        n_layers (int): The number of layers.

    Returns:
        None
    """
    layers_names = [f'Layer {i}' for i in range(n_layers)]
    labels = [f'(Pos {i}) {token_list[i]}' for i in range(len(token_list))]
    
    
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='coolwarm')
    
    ax.set_title('Patched Activation Matrix')
    ax.set_xticks(range(len(token_list)))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(layers_names)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Activation Difference', rotation=-90, va='bottom')
    
    # Centrar la imagen
    plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)

    if save:
        filename = ["".join(token_list[i].split()) for i in range(len(token_list)) if token_list[i] != '?']
        plt.savefig(f'./figures/{"".join(filename)}.pdf')
    plt.show()
    
frase = 5

if frase == 1:
    # FRASE 1
    prompt1 = "Michelle Jones was a top-notch student. Michelle"
    prompt2 = "Michelle Smith was a top-notch student. Michelle"
    generate_patched_activations(prompt1, prompt2, ['Smith', 'Jones'], num_samples=1, steps=1, do_sample=False)

elif frase == 2:
    # FRASE 2
    prompt1 = "Could you please help me with the homework?"
    prompt2 = "Should you please help me with the homework?"
    generate_patched_activations(prompt1, prompt2, ['Could', 'Should'], num_samples=1, steps=1, do_sample=False)

elif frase == 3:
    # FRASE 3
    prompt1 = "The book was on the table."
    prompt2 = "The book was under the table."
    generate_patched_activations(prompt1, prompt2, ['on', 'under'], num_samples=1, steps=1, do_sample=False)

elif frase == 4:
    # FRASE 4
    prompt1 = "The cat chased the mouse. The mouse was chased by the"
    prompt2 = "The dog chased the mouse. The mouse was chased by the"
    generate_patched_activations(prompt1, prompt2, ['cat', 'dog'], num_samples=1, steps=1, do_sample=False)

elif frase == 5:
    prompt1 = "Michelle Jones was a top-notch student. Michelle"
    prompt2 = "Jessica Jones was a top-notch student. Michelle"
    generate_patched_activations(prompt1, prompt2, ['Jessica', 'Michelle'], num_samples=1, steps=1, do_sample=False)

elif frase == 6:
    prompt1 = "Michelle Jones is a doctor. The doctor whose gender is"
    prompt2 = "Max Jones is a doctor. The doctor whose gender is"
    generate_patched_activations(prompt1, prompt2, ['Michelle', 'Max'], num_samples=1, steps=1, do_sample=False)

elif frase == 7:
    prompt1 = "The speaker's speech was inspiring and received a standing ovation."
    prompt2 = "The speaker's speech was disappointing and received a standing ovation."
    generate_patched_activations(prompt1, prompt2, ['inspiring', 'dissapointing'], num_samples=1, steps=1, do_sample=False)

elif frase == 8:
    prompt1 = "You will come to the party. You"
    prompt2 = "You will come to the party? You"
    generate_patched_activations(prompt1, prompt2, ['.', '?'], num_samples=1, steps=1, do_sample=False)