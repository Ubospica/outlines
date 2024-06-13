import json
from time import process_time_ns
from typing import Dict, List, Literal

from fastchat.conversation import get_conv_template
from pydantic import BaseModel

import outlines
from outlines import generate, models

model = models.transformers(
    # "/home/yxdong/dev/mlc-llm/dist/models/Llama-2-7b-chat-hf",
    "/home/yxdong/dev/mlc-llm/dist/models/Meta-Llama-3-8B-Instruct/",
    device="cuda",
    model_kwargs=dict(load_in_4bit=True),
    # model_kwargs=dict(load_in_8bit=True),
)

# class Product(BaseModel):
#     product_id: int
#     is_available: bool
#     price: float
#     is_featured: Literal[True]
#     category: Literal["Electronics", "Clothing", "Food"]
#     tags: List[str]
#     stock: Dict[str, int]


# schema_str = json.dumps(Product.model_json_schema())

# print("schema:", schema_str)

# system_prompt = (
#     "You are a helpful assistant. Always respond only with JSON based on the "
#     f"following JSON schema: {schema_str}."
# )
# prompt = (
#     "Generate a JSON that describes the product according to the given JSON schema."
# )


# # conv = get_conv_template("llama-2")
# conv = get_conv_template("llama-3")
# conv.set_system_message(system_prompt)
# conv.append_message(conv.roles[0], prompt)
# conv.append_message(conv.roles[1], None)
# full_prompt = conv.get_prompt()
# print(conv.get_prompt())

# batch = 1

# print("start preproc")
# start = process_time_ns()
# generator = generate.json(model, schema_str)
# end = process_time_ns()
# print("Preproc time:", (end - start) / 1e3)
# start = process_time_ns()
# end = process_time_ns()
# print("Empty time:", (end - start) / 1e3)
# answer = generator([full_prompt] * batch, max_tokens=500)

# print(answer)


# print("start preproc")
# start = process_time_ns()
# generator = generate.regex(
#     model,
#     r"""https:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"""
# )
# end = process_time_ns()
# print("Preproc time:", (end - start) / 1e3)
# prompt = "Generate the https link for a google website, with a path."
# answer = generator(prompt, max_tokens=500)

# print(answer)

# arithmetic_grammar = """
#     ?start: expression

#     ?expression: term (("+" | "-") term)*

#     ?term: factor (("*" | "/") factor)*

#     ?factor: NUMBER
#            | "-" factor
#            | "(" expression ")"

#     %import common.NUMBER
# """

# print("start preproc")
# start = process_time_ns()
# generator = generate.cfg(model, arithmetic_grammar)
# end = process_time_ns()
# print("Preproc time:", (end - start) / 1e3)
# sequence = generator(
#   "Alice had 4 apples and Bob ate 2. "
#   + "Write an expression for Alice's apples:"
# )

# print(sequence)


print("start preproc")
start = process_time_ns()
generator = generate.cfg(model, outlines.grammars.json)
end = process_time_ns()
print("Preproc time:", (end - start) / 1e3)
sequence = generator(
  "Generate a json object"
)

print(sequence)
