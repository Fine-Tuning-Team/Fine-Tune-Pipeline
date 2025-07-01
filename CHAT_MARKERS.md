# Chat Markers

To be used on `config.toml` for train_on_responses_only part.

## Qwen 2.5

```toml
    question_part = "<|im_start|>user\n"
    answer_part = "<|im_start|>assistant\n"
```

## Qwen 3

```toml
    question_part = "<|im_start|>user\n"
    answer_part = "<|im_start|>assistant\n"
```

## Gemma 3

```toml
    question_part = "<start_of_turn>user\n"
    answer_part = "<start_of_turn>model\n"
```

## Llama 3.2

Should work for 3.1 as well (I think).

```toml
    question_part = "<|start_header_id|>user<|end_header_id|>\n\n"
    answer_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Phi 4

```toml
    question_part="<|im_start|>user<|im_sep|>"
    answer_part="<|im_start|>assistant<|im_sep|>"
```

---

> INFO: You can find these markers formats in the unsloth [example notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) for specific models. Or else look for model-specific documentation in their respective repositories.
---
> If you can't find the marker format, you can use the `inferencer.py` script to generate a sample response and check the marker format used by the model. Or set the `train_on_responses_only` to `false` and run the fine-tuner to skip the marker format requirement. However, this might drop the model accuracy by a maximum of 10% (compared to training with the `train_on_responses_only` set to `true`).
