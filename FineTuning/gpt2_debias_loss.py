from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Tokenizer
import os
import warnings
from dataclasses import dataclass


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.modeling_gpt2 import GPT2DoubleHeadsModelOutput


class GPT2DoubleHeadsModelCustomLoss(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.debias_head = nn.functional.linear
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }

    # @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
            mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
                Index of the classification token in each input sequence.
                Selected in the range ``[0, input_ids.size(-1) - 1[``.
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
                Labels for language modeling.
                Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
                Indices are selected in ``[-1, 0, ..., config.vocab_size]``
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``
            mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
                Labels for computing the multiple choice classification loss.
                Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
                of the input tensors. (see `input_ids` above)
            kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
                Used to hide legacy arguments that have been deprecated.

        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "past" in kwargs:
            warnings.warn(
                "The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("past")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        print('lm_logits {}'.format(lm_logits))

        output_embed = self.get_output_embeddings()
        print('output_embed {}'.format(output_embed))

        print(output_embed.shape)

        target_embeds = self.transformer.get_input_embeddings()
        target_embeds = target_embeds(torch.LongTensor([12711, 4302]))
        print(target_embeds)
        debias_logits = self.debias_head(hidden_states, weight=target_embeds)
        print('debias_logits {}'.format(debias_logits))
        debias_logits = torch.squeeze(debias_logits)

        softmax_layer = nn.Softmax(dim=1)
        debias_softmax = softmax_layer(debias_logits)
        print('debias_softmax {}'.format(debias_softmax))

        # debias_logits = torch.squeeze(debias_logits)
        # print(debias_logits)
        # print(debias_logits[:, 0]/debias_logits[:, 1])
        #
        # debias_logits = torch.exp(debias_logits)
        # print('debias_logits exp {}'.format(debias_logits))

        debias_softmax = torch.squeeze(debias_softmax)
        debias_loss = torch.log(debias_softmax[:, 0]/debias_softmax[:, 1])

        debias_loss[torch.isnan(debias_loss)] = 0
        debias_loss = torch.sum(debias_loss)
        print('debias_loss {}'.format(debias_loss))

        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # lm_loss = lm_loss + debias_loss

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            if debias_loss is not None:
                output = (debias_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == '__main__':

    pretrained_model = 'microsoft/DialoGPT-small' #'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    model = GPT2DoubleHeadsModelCustomLoss.from_pretrained(pretrained_model, return_dict=False)

    num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    embedding_layer = model.resize_token_embeddings(len(tokenizer))

    # choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
    # encoded_choices = [tokenizer.encode(s) for s in choices]
    # cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
    # print(cls_token_location)

    input_ids = tokenizer.encode("They are very greedy and conspiring bunch of jews and they are so evil", return_tensors="pt")
    # mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

    # print(mc_token_ids)
    print(input_ids)
    outputs = model(input_ids, labels=input_ids)

    lm_loss = outputs[0]
    db_loss = outputs[1]

    print(lm_loss)
    print(db_loss)
    # print(outputs.loss)
    # print(outputs.mc_loss)