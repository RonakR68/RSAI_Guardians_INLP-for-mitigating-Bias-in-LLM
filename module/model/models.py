from functools import partial

import torch
import transformers

class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)

class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)


class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)

class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)


class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)


class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


