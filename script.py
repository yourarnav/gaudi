import os
import json
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Specify the model ID and number of accelerators
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
number_accelerators = 8  # Adjust this if needed

# Set HABANA_VISIBLE_DEVICES to specify the accelerators
os.environ["HABANA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Load the model using vLLM
model = LLM(model=model_id, tensor_parallel_size=number_accelerators, device='habana')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Set up the sampling parameters
sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=6500,
    n=20  # Number of generations per prompt
)

# Load the JSON file containing the product information and opinion summaries
with open("m3_50.json", "r") as file:
    products = json.load(file)

aspect_coverage_prompt_template = '''
user
Task Description:
You will be given a set of information such as product title, description, key features, specifications, reviews, and average rating. You will then be given one summary written for the set of information. Your task is to rate the summary on one metric. Make sure you understand the following evaluation metric very clearly. Your task is to rate the summary corresponding to the given set of information on the evaluation criteria.

Evaluation Criteria:
Aspect Coverage - Aspect Coverage measures how completely a summary captures the major features, characteristics, or attributes of a product that are prominently discussed in the original product information. Summaries should be penalized for missing any major aspects and rewarded for covering all important aspects thoroughly.

Following are the scores and the evaluation criteria according to which scores must be assigned.
<score>1</score> - Summary does not cover any important aspects present in the set of information.
<score>2</score> - Summary does not cover most of the important aspects present in the set of information.
<score>3</score> - Summary covers around half of the important aspects present in the set of information.
<score>4</score> - Summary covers most of the important aspects present in the set of information.
<score>5</score> - Summary covers all the important aspects discussed in the set of information.

Product Title: {product_title}

Description: {description}

Key Features: {key_features}

Specifications: {specifications}

Reviews: {reviews}

Average Rating: {average_rating}

Summary: {Product_Opinion_Summary}

Instructions:
Let's go step-by-step. Follow the following steps strictly while giving the response:

1. Identify the important aspects present in the set of information and list them with numbering.
2. Identify the important aspects present in the summary and list them with numbering.
3. Identify the important aspects covered by the summary that are present in the set of information and list them with numbering.
4. Calculate the total number of important aspects covered by the summary that are present in the set of information.
5. Calculate the total number of important aspects present in the set of information.
6. Finally use the evaluation criteria to output only a single score within <score></score> tags.

Note: Strictly give the score within <score></score> tags only e.g Score- <score>5</score>.

First give a detailed explanation of how much is the coverage and then finally give a single score following the format: Score- <score>5</score> assistant
'''

# Function to format specifications
def format_specifications(specifications):
    formatted_specs = []
    for spec in specifications:
        spec_str = f"{spec['key']}:\n"
        for value in spec['values']:
            spec_str += f"- {value['key']}: {value['value']}\n"
        formatted_specs.append(spec_str)
    return "\n".join(formatted_specs)

# Function to evaluate a single opinion summary
def evaluate_opinion_summary(product):
    product_title = product.get("product_title", "")
    description = product.get("description", "") if product.get("description") else "N/A"
    key_features = "\n".join(product.get("key_features", [])) if product.get("key_features") else "N/A"
    specifications = format_specifications(product.get("specifications", [])) if product.get("specifications") else "N/A"
    reviews = "\n".join(product.get("reviews", [])) if product.get("reviews") else "N/A"
    average_rating = product.get("averageRating", "N/A")
    opinion_summary = product.get("product_opinion_summary", "")

    prompt = aspect_coverage_prompt_template.format(
        product_title=product_title,
        description=description,
        key_features=key_features,
        specifications=specifications,
        reviews=reviews,
        average_rating=average_rating,
        Product_Opinion_Summary=opinion_summary
    )

    return prompt

# Process products in batches
batch_size = 5
evaluation_results = []

for i in range(0, len(products), batch_size):
    batch_products = products[i:i+batch_size]

    # Prepare prompts for the current batch
    prompts = [evaluate_opinion_summary(product) for product in batch_products]

    # Generate outputs using vLLM
    outputs = model.generate(prompts, sampling_params)

    # Process the outputs
    for j, output_group in enumerate(outputs):
        product_title = batch_products[j].get("product_title", "")
        for k, output in enumerate(output_group.outputs):
            generated_text = output.text
            evaluation_results.append(f"**{product_title} - Evaluation {k + 1}**\n\nResponse: {generated_text}\n")

    print(f"Processed batch {i // batch_size + 1} of {(len(products) - 1) // batch_size + 1}")

# Save the evaluation results to a text file
with open("m3_ac_tj_dep.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(evaluation_results))
