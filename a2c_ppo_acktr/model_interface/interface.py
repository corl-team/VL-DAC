import torch
import math
from transformers import GenerationConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def model_generate(
    value_model,
    processor,
    input_ids,
    token_type_ids=None,
    pixel_values=None,
    image_grid_thw=None,
    pixel_values_videos=None,
    video_grid_thw=None,
    attention_mask=None,
    args=None,
):
    """
    Generate text using the Qwen2VL model with the given inputs.

    Args:
        model (Qwen2VLModel): The Qwen2VL model.
        tokenizer (Qwen2VLTokenizer): The corresponding tokenizer.
        input_ids (torch.Tensor): Tokenized input IDs.
        pixel_values (torch.Tensor): Preprocessed image tensor.
        image_grid_thw (torch.Tensor): Grid size for image.
        pixel_values_videos (torch.Tensor): Preprocessed video tensor.
        video_grid_thw (torch.Tensor): Grid size for video.
        args: Generation arguments including temperature, num_beams, max_new_tokens, etc.

    Returns:
        Tuple containing:
            - values: Computed values from the value head.
            - padded_output_ids: Padded generated token IDs.
            - outputs: Decoded output strings.
            - sum_log_probs: Sum of log probabilities.
            - action_tokens_log_prob: Log probabilities of action tokens.
    """
    base_model = value_model.base_model
    device = base_model.device
    if not (pixel_values is None):
        pixel_values = pixel_values.to(device)
    if not (image_grid_thw is None):
        image_grid_thw = image_grid_thw.to(device)
    if not (pixel_values_videos is None):
        pixel_values_videos = pixel_values_videos.to(device)
    if not (video_grid_thw is None):
        video_grid_thw = video_grid_thw.to(device)
    if not (token_type_ids is None):
        token_type_ids = token_type_ids.to(device)

    # Prepare inputs for Qwen2VL
    if token_type_ids is not None:
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "token_type_ids": token_type_ids,
        }
    else:
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=True,
        temperature=args.temperature,
        repetition_penalty=1.0, 
        top_p=0.001,
        top_k=1,
    )
    with torch.inference_mode():
        outputs = base_model.generate(
            **inputs,
            generation_config=generation_config,
            output_scores=True,
            output_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    output_ids = outputs.sequences

    # Decode the generated tokens
    generated_ids_trimmed = output_ids[:, input_ids.size(1) :]
    decoded_outputs = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )
    # Pad the output_ids to a fixed length
    # padded_output_ids = torch.zeros(
    #     generated_ids_trimmed.size(0),
    #     2 * args.max_new_tokens,
    #     dtype=output_ids.dtype,
    #     device=output_ids.device,
    # )
    # padded_output_ids[:, : generated_ids_trimmed.size(1)] = generated_ids_trimmed

    # Evaluate the generated outputs
    
    values, tokens_log_probs = model_evaluate(
        **inputs,
        value_model=value_model,
        processor=processor,
        output_ids=generated_ids_trimmed,
        temperature=args.temperature,
        thought_prob_coef=args.thought_prob_coef,
        value_stopgrad=args.stop_grad,
    )

    return (
        values,
        generated_ids_trimmed,
        decoded_outputs,
        tokens_log_probs,
    )


def model_evaluate(
    value_model,
    processor,
    input_ids,
    output_ids,
    token_type_ids=None,
    pixel_values=None,
    image_grid_thw=None,
    pixel_values_videos=None,
    video_grid_thw=None,
    attention_mask=None,
    temperature=None,
    thought_prob_coef=None,
    mode="eval", 
    value_stopgrad=True,

):
    """
    Evaluate the generated outputs using the Qwen2VL model.

    Args:
        model (Qwen2VLModel): The Qwen2VL model.
        tokenizer (Qwen2VLTokenizer): The corresponding tokenizer.
        input_ids (torch.Tensor): Original input token IDs.
        output_ids (torch.Tensor): Generated output token IDs.
        pixel_values (torch.Tensor): Preprocessed image tensor.
        image_grid_thw (torch.Tensor): Grid size for image.
        pixel_values_videos (torch.Tensor): Preprocessed video tensor.
        video_grid_thw (torch.Tensor): Grid size for video.
        temperature (float): Temperature parameter for scaling logits.
        thought_prob_coef (float): Coefficient for thought log probabilities.

    Returns:
        Tuple containing:
            - values: Computed values from the value head.
            - sum_log_prob: Combined sum of log probabilities.
            - action_tokens_log_prob: Log probabilities of action tokens.
    """
    base_model = value_model.base_model
    device = base_model.device
    dtype = base_model.dtype

    if not (pixel_values is None):
        pixel_values = pixel_values.to(device)
    if not (image_grid_thw is None):
        image_grid_thw = image_grid_thw.to(device)
    if not (pixel_values_videos is None):
        pixel_values_videos = pixel_values_videos.to(device)
    if not (video_grid_thw is None):
        video_grid_thw = video_grid_thw.to(device)
    if not (token_type_ids is None):
        token_type_ids = token_type_ids.to(device)
    # Broadcast input_ids if batch size > 1
    if output_ids.size(0) != 1:
        input_ids = input_ids.expand(output_ids.size(0), -1)
    input_ids = input_ids.to(device)
    output_ids = output_ids.to(device)
    if not (token_type_ids is None):
        token_type_ids = torch.cat([token_type_ids, torch.zeros_like(output_ids, device=device)], dim=1)
    combined_ids = torch.cat([input_ids, output_ids], dim=1).to(device)

    # Expand attention_mask to match the full size of combined_ids
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.size(0) == 1 and output_ids.size(0) > 1:
            attention_mask = attention_mask.expand(output_ids.size(0), -1)
        # Create a mask for the output_ids (assuming no padding tokens)
        output_mask = torch.ones_like(output_ids, dtype=attention_mask.dtype, device=device)
        # Concatenate the input and output masks
        attention_mask = torch.cat([attention_mask, output_mask], dim=1)
    else:
        # If attention_mask is None, create a mask of ones for the combined_ids
        attention_mask = torch.ones_like(combined_ids, dtype=torch.long, device=device)

    # Prepare inputs for evaluation
    if token_type_ids is not None:
        inputs = {
            "input_ids": combined_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "token_type_ids": token_type_ids,
            # "attention_mask": attention_mask,
        }
    else:
        inputs = {
            "input_ids": combined_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
    # for key, item in inputs.items(): print(key, item.dtype) if item is not None else None

    outputs = base_model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]

    if mode == "eval":
        logits = logits.detach()
        hidden_states = hidden_states.detach()
    
    # Compute values from the value head (assuming the model has a value head)
    # If not, this part should be adjusted accordingly
    if hasattr(value_model, "value_head"):
        input_token_len = input_ids.size(1)
        if value_stopgrad:
            hidden_state_to_value = hidden_states[:, input_token_len - 1].detach()
        else:
            hidden_state_to_value = hidden_states[:, input_token_len - 1]
        values = value_model.value_head(hidden_state_to_value)
        if mode == "eval":
            values = values.detach()
    else:
        raise AttributeError("The model does not have a 'value_head' attribute.")

    # Scale logits by temperature and compute log probabilities
    scores = logits#s * (1.0 / temperature)
    scores = scores.to(torch.float32)
    
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1).to(torch.bfloat16)
    # assert not inv_scores.requires_grad, "inv_scores should not require gradients"
    # assert log_probs.requires_grad, "log_probs should require gradients"
    # inv_scores = (1.0 - scores).to(torch.bfloat16)
    # log_probs = torch.log(scores).to(torch.bfloat16)
    # Create mask for valid output tokens
    output_ids_mask = (output_ids != 0)[:, 1:]
    tokens_log_probs = output_ids_mask * torch.gather(
        log_probs[:, input_ids.size(1) : -1], 2, output_ids[:, 1:].unsqueeze(2)
    ).squeeze(2)

    # Define target tokens for 'action' (adjust the token IDs as per tokenizer)
    
    return values, tokens_log_probs

def model_evaluate_reference(
    model,
    processor,
    input_ids,
    output_ids,
    token_type_ids=None,
    pixel_values=None,
    image_grid_thw=None,
    pixel_values_videos=None,
    video_grid_thw=None,
    attention_mask=None,
    temperature=None,
    thought_prob_coef=None,
    mode="eval"
):
    """
    Evaluate the generated outputs using the Qwen2VL model.

    Args:
        model (Qwen2VLModel): The Qwen2VL model.
        tokenizer (Qwen2VLTokenizer): The corresponding tokenizer.
        input_ids (torch.Tensor): Original input token IDs.
        output_ids (torch.Tensor): Generated output token IDs.
        pixel_values (torch.Tensor): Preprocessed image tensor.
        image_grid_thw (torch.Tensor): Grid size for image.
        pixel_values_videos (torch.Tensor): Preprocessed video tensor.
        video_grid_thw (torch.Tensor): Grid size for video.
        temperature (float): Temperature parameter for scaling logits.
        thought_prob_coef (float): Coefficient for thought log probabilities.

    Returns:
        Tuple containing:
            - values: Computed values from the value head.
            - sum_log_prob: Combined sum of log probabilities.
            - action_tokens_log_prob: Log probabilities of action tokens.
    """
    base_model = model
    device = base_model.device
    dtype = base_model.dtype

    if not (pixel_values is None):
        pixel_values = pixel_values.to(device)
    if not (image_grid_thw is None):
        image_grid_thw = image_grid_thw.to(device)
    if not (pixel_values_videos is None):
        pixel_values_videos = pixel_values_videos.to(device)
    if not (video_grid_thw is None):
        video_grid_thw = video_grid_thw.to(device)
    if not (token_type_ids is None):
        token_type_ids = token_type_ids.to(device)

    # Broadcast input_ids if batch size > 1
    if output_ids.size(0) != 1:
        input_ids = input_ids.expand(output_ids.size(0), -1)
    input_ids = input_ids.to(device)
    output_ids = output_ids.to(device)
    if not (token_type_ids is None):
        token_type_ids = torch.cat([token_type_ids, torch.zeros_like(output_ids, device=device)], dim=1)
    combined_ids = torch.cat([input_ids, output_ids], dim=1).to(device)

    # Expand attention_mask to match the full size of combined_ids
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.size(0) == 1 and output_ids.size(0) > 1:
            attention_mask = attention_mask.expand(output_ids.size(0), -1)
        # Create a mask for the output_ids (assuming no padding tokens)
        output_mask = torch.ones_like(output_ids, dtype=attention_mask.dtype, device=device)
        # Concatenate the input and output masks
        attention_mask = torch.cat([attention_mask, output_mask], dim=1)
    else:
        # If attention_mask is None, create a mask of ones for the combined_ids
        attention_mask = torch.ones_like(combined_ids, dtype=torch.long, device=device)

    # Prepare inputs for evaluation
    if token_type_ids is not None:
        inputs = {
            "input_ids": combined_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "token_type_ids": token_type_ids,
            # "attention_mask": attention_mask,
        }
    else:
        inputs = {
            "input_ids": combined_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
    # for key, item in inputs.items(): print(key, item.dtype) if item is not None else None

    outputs = base_model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]

    if mode == "eval":
        logits = logits.detach()
        hidden_states = hidden_states.detach()

    # Scale logits by temperature and compute log probabilities
    scores = logits#  * (1.0 / temperature)
    scores = scores.to(torch.float32)
    
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1).to(torch.bfloat16)
    # assert not inv_scores.requires_grad, "inv_scores should not require gradients"
    # assert log_probs.requires_grad, "log_probs should require gradients"
    # inv_scores = (1.0 - scores).to(torch.bfloat16)
    # log_probs = torch.log(scores).to(torch.bfloat16)
    # Create mask for valid output tokens
    output_ids_mask = (output_ids != 0)[:, 1:]
    tokens_log_probs = output_ids_mask * torch.gather(
        log_probs[:, input_ids.size(1) : -1], 2, output_ids[:, 1:].unsqueeze(2)
    ).squeeze(2)

    # Define target tokens for 'action' (adjust the token IDs as per tokenizer)
    
    return tokens_log_probs